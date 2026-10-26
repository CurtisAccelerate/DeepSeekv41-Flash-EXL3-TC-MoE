#define EXL3_MOE_COOP_DEFINE_ROT
#include "exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_a_kernel<2, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_b_kernel<2, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_a_kernel<3, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_b_kernel<3, 2, true, false>(const MoeCoopParams);
