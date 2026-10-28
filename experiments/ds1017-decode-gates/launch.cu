// Serving launcher for the 2-6 user gate libraries.
// GU_DIRECT stays false in that build, so gate/up keeps the FP16 fold.
// KBITS is 3 for libgu_rows.so and 2 for libgu_k2.so.
// Six rows stay on the mixed-attention kernel. Forty-two rows are not captured.
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#ifndef GU_DIRECT
#define GU_DIRECT false
#endif
#ifndef KBITS
#define KBITS 3
#endif
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"

struct GuLaunch {
    void* x;
    void* sel;
    void* rw;
    void* g_trellis;
    void* g_suh;
    void* g_svh;
    void* u_trellis;
    void* u_suh;
    void* u_svh;
    void* d_trellis;
    void* d_suh;
    void* d_svh;
    void* had_g;
    void* had_u;
    void* gu_g;
    void* gu_u;
    void* act_out;
    void* d_out;
    void* ctr;
    void* out;
    int64_t x_stride;
    int64_t out_stride;
    int64_t bsz;
    int64_t topk;
    int64_t H;
    int64_t Hi;
    int64_t I;
    int64_t Ho;
    int64_t H_out;
    int64_t min_expert;
    int64_t max_expert;
    int64_t act;
    int64_t gated;
    int64_t gu_f32;
    int64_t slots_max;
    int64_t rows_max;
    int64_t n_local;
    double act_limit;
};

static void ck(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "gu_launch %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
}

template <int SPEC>
static void gu_launch_spec(const GuLaunch* in, cudaStream_t stream) {
    MoeCoopParams p = {};
    p.x = reinterpret_cast<const half*>(in->x);
    p.x_stride = (int) in->x_stride;
    p.sel = reinterpret_cast<const int64_t*>(in->sel);
    p.rw = reinterpret_cast<const half*>(in->rw);
    p.bsz = (int) in->bsz;
    p.topk = (int) in->topk;
    p.H = (int) in->H;
    p.Hi = (int) in->Hi;
    p.I = (int) in->I;
    p.Ho = (int) in->Ho;
    p.H_out = (int) in->H_out;
    p.min_expert = (int) in->min_expert;
    p.max_expert = (int) in->max_expert;
    p.g_trellis = reinterpret_cast<const int64_t*>(in->g_trellis);
    p.g_suh = reinterpret_cast<const int64_t*>(in->g_suh);
    p.g_svh = reinterpret_cast<const int64_t*>(in->g_svh);
    p.u_trellis = reinterpret_cast<const int64_t*>(in->u_trellis);
    p.u_suh = reinterpret_cast<const int64_t*>(in->u_suh);
    p.u_svh = reinterpret_cast<const int64_t*>(in->u_svh);
    p.d_trellis = reinterpret_cast<const int64_t*>(in->d_trellis);
    p.d_suh = reinterpret_cast<const int64_t*>(in->d_suh);
    p.d_svh = reinterpret_cast<const int64_t*>(in->d_svh);
    p.g_bias = nullptr;
    p.u_bias = nullptr;
    p.d_bias = nullptr;
    p.act = (int) in->act;
    p.act_limit = (float) in->act_limit;
    p.gated = in->gated != 0;
    p.had_g = reinterpret_cast<half*>(in->had_g);
    p.had_u = reinterpret_cast<half*>(in->had_u);
    p.a_global = p.bsz > 1;
    p.gu_g = in->gu_g;
    p.gu_u = in->gu_u;
    p.gu_f32 = in->gu_f32 != 0;
    p.act_out = reinterpret_cast<half*>(in->act_out);
    p.d_out = reinterpret_cast<float*>(in->d_out);
    p.slots_max = (int) in->slots_max;
    p.rows_max = (int) in->rows_max;
    p.n_local = (int) in->n_local;
    p.ctr_a_len = p.slots_max * (p.I / 128);
    p.ctr_b_len = p.rows_max * (p.Ho / 128);
    p.ctr_a = reinterpret_cast<int*>(in->ctr);
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.ksplit_a = 1;
    p.ksplit_b = 1;
    p.dbg = 0;
    p.sh_gate_n = 0;
    p.sh_out = nullptr;
    p.sh_gate_w = nullptr;
    p.out = reinterpret_cast<float*>(in->out);
    p.out_stride = (int) in->out_stride;

    const int slots = p.bsz * p.topk;
    const int nproj = p.gated ? 2 : 1;
    const int grid_a = slots * nproj * (p.I / 128);
    const int grid_b = slots * (p.Ho / 128);
    const int smem_a = exl3_moe_coop_ns::smem_a_bytes<KBITS>(p.Hi);
    const int smem_b = exl3_moe_coop_ns::smem_b_bytes<KBITS>();
    auto* ka = exl3_moe_coop_ns::exl3_moe_coop_a_kernel<KBITS, 2, true, false, SPEC>;
    auto* kb = exl3_moe_coop_ns::exl3_moe_coop_b_kernel<KBITS, 2, true, false, SPEC>;
    if (smem_a > 48 * 1024)
        ck(cudaFuncSetAttribute(ka, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_a), "smem-a");
    if (smem_b > 48 * 1024)
        ck(cudaFuncSetAttribute(kb, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_b), "smem-b");

    if (p.a_global) {
        const int rot_items = slots * (p.Hi / 128) * nproj;
        const int rot_grid = (rot_items + 15) / 16;
        exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<rot_grid, 512, 0, stream>>>(p);
    }
    ka<<<grid_a, 512, smem_a, stream>>>(p);
    kb<<<grid_b, 512, smem_b, stream>>>(p);
    ck(cudaGetLastError(), "kernels");
}

extern "C" void gu_launch(const GuLaunch* in, void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    switch ((int) in->bsz) {
        case 8: gu_launch_spec<8>(in, stream); break;
        case 6: gu_launch_spec<6>(in, stream); break;
        case 12: gu_launch_spec<12>(in, stream); break;
        case 18: gu_launch_spec<18>(in, stream); break;
        case 24: gu_launch_spec<24>(in, stream); break;
        case 30: gu_launch_spec<30>(in, stream); break;
        case 36: gu_launch_spec<36>(in, stream); break;
        case 42: gu_launch_spec<42>(in, stream); break;
        default:
            std::fprintf(stderr, "gu_launch: rows %lld are outside 6,8,12,18,24,30,36,42\n", (long long) in->bsz);
            std::exit(1);
    }
}
