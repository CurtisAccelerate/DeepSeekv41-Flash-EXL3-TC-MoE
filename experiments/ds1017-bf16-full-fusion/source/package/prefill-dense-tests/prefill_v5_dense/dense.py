"""Exact-weight dense prefill runner. No torch import or GPU work on import."""
ROWS = (65, 128, 256, 512, 1024, 2032, 2048)
MIN_ELEMENTS = 4 << 20
MAX_ELEMENTS = 32 << 20


def tensor_pin(t):
    return (int(t.data_ptr()), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


def geometry(layer):
    return int(layer.output_size_per_partition), int(layer.input_size_per_partition)


def workspace_elements(N, K):
    if N % 16 or K % 64 or not MIN_ELEMENTS < N * K <= MAX_ELEMENTS:
        raise ValueError('Dense geometry outside retained native-pack domain')
    return N * K // 2  # float32 elements: same bytes as N*K BF16 elements.


class DenseRunner:
    def __init__(self, torch, workspace, device, ext, marlin):
        self.torch, self.workspace, self.device = torch, workspace, device
        self.ext, self.marlin = ext, marlin
        self.stream = int(torch.cuda.current_stream(device).cuda_stream)
        self.ws_pin = tensor_pin(workspace)
        self.bindings = {}
        self.qualified = set()
        self.tuned = set()
        self.busy = False

    def bind(self, name, layer):
        N, K = geometry(layer)
        needed = workspace_elements(N, K)
        if len(layer._abc_w8) != 3:
            raise ValueError('Actual retained native pack must be (W,S,cnt)')
        if needed > self.workspace.numel():
            raise ValueError('Existing shared workspace is too small; no new WB allocation allowed')
        packed = tuple(layer._abc_w8)
        kept = (layer.weight, layer.weight_scale, layer.workspace)
        if any(t.device != self.device for t in packed + kept):
            raise ValueError('Dense binding spans devices')
        # Every view aliases the SAME workspace prefix; only stream-ordered calls
        # are allowed. No clone/empty/contiguous allocation for BF16 weights.
        wb = self.workspace[:needed].view(self.torch.bfloat16).view(N, K)
        if wb.data_ptr() != self.workspace.data_ptr() or wb.numel() * wb.element_size() != 2 * N * K:
            raise ValueError('Dense BF16 weight view does not alias exact workspace bytes')
        self.bindings[id(layer)] = dict(name=name, layer=layer, N=N, K=K,
                                      packed=packed, kept=kept, wb=wb,
                                      pins=tuple(tensor_pin(t) for t in packed + kept))

    def _binding(self, layer):
        b = self.bindings.get(id(layer))
        if b is None:
            return None
        if geometry(layer) != (b['N'], b['K']) or tensor_pin(self.workspace) != self.ws_pin:
            raise RuntimeError('Prepared dense geometry/workspace moved')
        tensors = tuple(layer._abc_w8) + (layer.weight, layer.weight_scale, layer.workspace)
        if len(tensors) != 6 or tuple(tensor_pin(t) for t in tensors) != b['pins']:
            raise RuntimeError('Prepared dense pack/Marlin pointers changed')
        return b

    def eligible(self, layer, x, bias):
        t = self.torch
        if bias is not None or getattr(layer, 'bias', None) is not None:
            return False
        b = self._binding(layer)
        if b is None or x.dtype != t.bfloat16 or x.device != self.device:
            return False
        if t.cuda.is_current_stream_capturing() or int(t.cuda.current_stream(self.device).cuda_stream) != self.stream:
            return False
        return (x.ndim >= 2 and x.shape[-1] == b['K'] and x.is_contiguous()
                and x.data_ptr() % 16 == 0 and 65 <= x.numel() // b['K'] <= 2048)

    def _compute(self, b, x):
        x2 = x.reshape(-1, b['K'])
        self.ext.dequant(b['packed'][0], b['packed'][1], b['wb'], b['N'], b['K'])
        backend = self.torch.backends.cuda.matmul
        previous = backend.allow_bf16_reduced_precision_reduction
        try:
            backend.allow_bf16_reduced_precision_reduction = False
            out = self.torch.mm(x2, b['wb'].t())
        finally:
            backend.allow_bf16_reduced_precision_reduction = previous
        return out.reshape(*x.shape[:-1], b['N'])

    def reference(self, layer, x):
        b = self._binding(layer)
        if b is None:
            raise ValueError('No retained Marlin binding')
        return self.marlin(input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                           workspace=layer.workspace, size_n=b['N'], size_k=b['K'], bias=None)

    def apply(self, layer, x, policy='cublas'):
        if policy not in ('cublas', 'tuned'):
            raise ValueError('Unknown dense policy')
        if not self.eligible(layer, x, None):
            return None
        b = self._binding(layer)
        key = (b['N'], b['K'], x.numel() // b['K'])
        if not all((b['N'], b['K'], rows) in self.qualified for rows in ROWS):
            return None
        if policy == 'tuned' and key not in self.tuned:
            return None
        if self.busy:
            raise RuntimeError('Reentrant shared dense workspace use')
        self.busy = True
        try:
            return self._compute(b, x)
        finally:
            self.busy = False


def prepare(torch, model, workspace, device):
    """Explicit GPU startup work; reject only this component on ordinary failure.

    Exceptions propagate to parent, which must distinguish CUDA health failures
    from memory/numeric rejection without disabling independent candidates.
    """
    import dsv41_fp8w_ext as ext
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_mxfp8_marlin_linear
    from hybrid import abc_numeric
    from .qualification import qualify
    from .source_contract import check_runtime
    source_contract = check_runtime(apply_mxfp8_marlin_linear, abc_numeric)
    device = torch.device(device)
    if (workspace.dtype != torch.float32 or workspace.device != device
            or not workspace.is_contiguous() or workspace.ndim != 1 or workspace.data_ptr() % 16):
        raise ValueError('Dense scratch must be the existing aligned 1-D FP32 shared workspace')
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('Dense preparation must be eager')
    runner = DenseRunner(torch, workspace, device, ext, apply_mxfp8_marlin_linear)
    counts = {}
    for name, layer in model.named_modules():
        if not hasattr(layer, '_abc_w8'):
            continue
        N, K = geometry(layer)
        if not MIN_ELEMENTS < N * K <= MAX_ELEMENTS or N % 16 or K % 64 or getattr(layer, 'bias', None) is not None:
            continue
        runner.bind(name, layer)
        counts[(N, K)] = counts.get((N, K), 0) + 1
    if not counts:
        raise ValueError('No eligible actual retained native dense layers')
    # Existing prepare_dense checks EVERY source pack before storing _abc_w8.
    # The actual receipt list must prove enough successful checks per geometry.
    for key, count in counts.items():
        records = [r for r in abc_numeric.DENSE if (r.get('N'), r.get('K')) == key]
        if len(records) < count or not all(r.get('source_weight_bits_exact') is True for r in records):
            raise ValueError('Missing actual abc_numeric.DENSE source-bit-exact records: ' + str(key))
    receipt = qualify(torch, runner, device)
    receipt.update(layers=len(runner.bindings), geometries=len(counts),
                   source_weight_records_asserted=True, shared_workspace_ptr=int(workspace.data_ptr()),
                   allocated_weight_buffers=0, model_quality_verified=False, source_contract=source_contract)
    return runner, receipt
