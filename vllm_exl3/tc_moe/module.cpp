// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Codec adapted from turboderp-org/exllamav3 (MIT); the MIT grant is in codec.cuh.

#include "moe_prefill.h"
#include "moe_v2.h"

#include <Python.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/version.h>

// TORCH_LIBRARY is not the stable ABI. This library loads only into the
// PyTorch it was compiled against.

TORCH_LIBRARY(_dsv41_tc_moe_C, m) {
  m.def("moe_out_v2(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, int bits_x2, float limit, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("moe_out_prefill(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, int bits_x2, float limit, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("moe_workspace_size_v2(int max_rows, int hidden, int intermediate, "
        "int topk, int local_experts) -> int",
        &exl3::moe_workspace_size_v2);
  m.def("moe_workspace_size_prefill(int max_rows, int hidden, int intermediate, "
        "int topk, int local_experts) -> int",
        &exl3::moe_workspace_size_prefill);
}

TORCH_LIBRARY_IMPL(_dsv41_tc_moe_C, CUDA, m) {
  m.impl("moe_out_v2", &exl3::moe_out_v2);
  m.impl("moe_out_prefill", &exl3::moe_out_prefill);
}

namespace {

PyObject* build_info(PyObject*, PyObject*) {
  return Py_BuildValue("{s:i,s:s,s:i,s:i}", "abi_version", 1, "torch_version",
                       TORCH_VERSION, "cuda_version", CUDART_VERSION,
                       "cxx11_abi", _GLIBCXX_USE_CXX11_ABI);
}

PyMethodDef methods[] = {
    {"build_info", build_info, METH_NOARGS, "Return the compiled ABI metadata."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

PyMODINIT_FUNC PyInit__dsv41_tc_moe_C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_dsv41_tc_moe_C", nullptr, 0,
                               methods};
  return PyModule_Create(&module);
}
