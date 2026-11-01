"""Runtime Python API checks only, before any dense qualification allocation."""
import inspect

MARLIN_ARGS = ['input', 'weight', 'weight_scale', 'workspace', 'size_n',
               'size_k', 'bias', 'use_fp32_reduce']


def check_runtime(marlin, numeric):
    if list(inspect.signature(marlin).parameters) != MARLIN_ARGS:
        raise ValueError('Retained Marlin helper API changed')
    if not isinstance(getattr(numeric, 'DENSE', None), list):
        raise ValueError('Actual abc_numeric.DENSE receipt list unavailable')
    return {'passed': True, 'marlin_args': MARLIN_ARGS, 'receipt_field': 'source_weight_bits_exact'}
