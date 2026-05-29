#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "quants.h"
#include "math_fp.h"

// Q4_0 x Q8_0 -> F32 MUL_MAT on the tensor unit using INT8 TensorIMA8A32.
//
// Activations are pre-quantized to Q8_0 by the separate quantize_mat_q8_0 kernel
// (src1 is Q8_0 here). Weights (Q4_0) -> int8 (nibble-8) with per-block fp16
// scale d_w. The Q8_0 activations give int8 qs + per-block scale scale_a.
// Per 32-elem K block: TensorIMA8A32 (int8xint8->int32 in TenC, copied to FREGS),
// then tensor_quant scales int32->fp32 by d_w (per weight col m) and scale_a (per
// activation row n), then VPU fadd.ps accumulates into a separate FP32 reg block.
//
// Dual-hart (like mul_mat_f16_matrix_engine.c): hart 1 produces int8 panels +
// scale vectors into double-buffered L2 SCP; hart 0 runs the tensor pipeline.
// v1: scalar quant/unpack (correctness first); k_splits=1.
//
// TILE_N=8 so the int32/FP32 block result (N*M=128) AND the FP32 accumulator
// (128) both fit in the 32x256-bit vector register file (256 fp32 total).

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  8
#define BLOCK_K QK4_0   // 32 elements per Q4_0 block

#define A_L1_START 0    // L1 SCP lines 0..7  : A (int8 activations, plain)
#define B_L1_START 8    // L1 SCP lines 8..15 : B (int8 weights, interleave8)
#define S_L1_START 16   // L1 SCP lines 16..17: scale vectors (d_w, scale_a)

// FREG layout: block result rows -> f0..f(2*TILE_N-1); accumulator -> f16..f31.
#define ACC_REG_START 16

// L2 SCP panel layout per minion (double-buffered).
//   A int8:  TILE_N rows x 64 B            = 512 B
//   B int8:  32 k-rows x 16 m (row-major)  = 512 B  (-> Interleave8 on load)
//   d_w:     16 f32                        = 64 B
//   scale_a: 16 f32                        = 64 B   (TILE_N used)
#define PANEL_A_OFF   0
#define PANEL_B_OFF   512
#define PANEL_DW_OFF  1024
#define PANEL_SA_OFF  1088
#define PANEL_BYTES   1152                                  // per buffer
#define SCP_READY_OFF    (2 * PANEL_BYTES)                  // 2304
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)               // 2368
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)            // 2432

static inline void __attribute__((always_inline))
scp_signal(volatile uint32_t *flag, uint32_t value) {
    *flag = value;
    FENCE;
    evict_to_l2((const void *)flag, 1, 64);
    WAIT_CACHEOPS;
}

static inline void __attribute__((always_inline))
scp_wait(volatile uint32_t *flag, uint32_t expected) {
    while (1) {
        evict_to_l2((const void *)flag, 1, 64);
        WAIT_CACHEOPS;
        if (*flag >= expected) return;
    }
}

// Read n_cur pre-quantized Q8_0 activation columns for K-block kb: copy int8 qs
// into the A panel (row n -> 64-byte line) and extract scale_a[n] = fp16(d).
static inline void __attribute__((always_inline))
read_act_block_q8_0(int8_t *a_panel, float *scale_a, const char *src1_batch,
                    int64_t nb, int64_t kb, int64_t n_cur, int64_t nb1_1) {
    for (int n = 0; n < n_cur; ++n) {
        const block_q8_0 *blk =
            (const block_q8_0 *)(src1_batch + (nb + n) * nb1_1) + kb;
        scale_a[n] = fp16_to_fp32(blk->d);
        int8_t *q = a_panel + (int64_t) n * 64;
        for (int k = 0; k < QK8_0; ++k) {
            q[k] = blk->qs[k];
        }
    }
}

// Unpack 16 Q4_0 weight rows (one 32-elem block) to int8 (nibble-8) directly in
// the interleaved TenB layout that TensorIMA8A32 consumes: for k-pass line i,
// byte (m*4 + r) holds weight w[4i+r][m]  (k = 4i+r). 32 K -> 8 lines x 64 B.
// Also fills d_w[m]. Loaded via use_tenb=1 (plain -> TenB buffer), like power.c.
static inline void __attribute__((always_inline))
unpack_weight_block(int8_t *b_panel, float *d_w, const char *src0_batch,
                    int64_t mb, int64_t kb_block, int64_t nb1_0) {
    for (int m = 0; m < TILE_M; ++m) {
        const block_q4_0 *blk =
            (const block_q4_0 *)(src0_batch + (mb + m) * nb1_0) + kb_block;
        d_w[m] = fp16_to_fp32(blk->d);
        for (int i = 0; i < QK4_0 / 2; ++i) {     // 16 packed bytes
            const int lo = (int)(blk->qs[i] & 0xF) - 8;   // k = i
            const int hi = (int)(blk->qs[i] >> 4)  - 8;   // k = i + 16
            // k=i      -> line i/4,     lane r=i%4
            b_panel[(i / 4) * 64 + m * 4 + (i % 4)]         = (int8_t) lo;
            // k=i+16   -> line i/4 + 4, lane r=i%4
            b_panel[((i / 4) + 4) * 64 + m * 4 + (i % 4)]   = (int8_t) hi;
        }
    }
}

int entry_point(struct ggml_et_binary_params *params, void *env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();
    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int is_hart1 = hart_id & 1;
    uint64_t local_minion = (hart_id >> 1) & 0x1F;

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    if ((M % TILE_M) != 0)  return 0;
    if ((K % BLOCK_K) != 0) return 0;

    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];
    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    const int64_t nb1_1 = params->src1.nb[1];
    const int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    const int64_t nb1_d = params->dst.nb[1];
    const int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    const char *src0_base = (const char *) params->src0.data;
    const char *src1_base = (const char *) params->src1.data;
    char       *dst_base  = (char *) params->dst.data;

    const int64_t m_tiles = M / TILE_M;
    const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
    const int64_t batch_count = ne2_1 * ne3_1;
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;

    const int64_t k_steps = K / BLOCK_K;
    const int64_t k_splits = 1;                 // v1: single K-split
    const int64_t tiles_per_shire = MINIONS_PER_SHIRE;
    const int64_t local_tile_idx = local_minion;
    const int64_t tiles_stride = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;
    const int64_t kb_start = 0;
    const int64_t kb_end   = k_steps;
    (void) k_splits;

    uint64_t scp_base = local_minion * SCP_PER_MINION;
    int8_t *a_panel[2] = {
        (int8_t *) et_shire_l2scp_local(scp_base + PANEL_A_OFF),
        (int8_t *) et_shire_l2scp_local(scp_base + PANEL_BYTES + PANEL_A_OFF),
    };
    int8_t *b_panel[2] = {
        (int8_t *) et_shire_l2scp_local(scp_base + PANEL_B_OFF),
        (int8_t *) et_shire_l2scp_local(scp_base + PANEL_BYTES + PANEL_B_OFF),
    };
    float *dw_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base + PANEL_DW_OFF),
        (float *) et_shire_l2scp_local(scp_base + PANEL_BYTES + PANEL_DW_OFF),
    };
    float *sa_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base + PANEL_SA_OFF),
        (float *) et_shire_l2scp_local(scp_base + PANEL_BYTES + PANEL_SA_OFF),
    };
    volatile uint32_t *ready_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    volatile uint32_t *consumed_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: int8 producer (activation quant + weight unpack + scales)
    // ================================================================
    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;
        for (int64_t tile = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
             tile < base_tiles; tile += tiles_stride) {

            const int64_t tiles_per_batch = m_tiles * n_tiles;
            const int64_t batch_idx       = tile / tiles_per_batch;
            const int64_t tile_in_batch   = tile % tiles_per_batch;
            const int64_t nb_idx = tile_in_batch / m_tiles;
            const int64_t mb_idx = tile_in_batch % m_tiles;

            const int64_t i3   = batch_idx / ne2_1;
            const int64_t i2   = batch_idx % ne2_1;
            const int64_t i2_0 = i2 / r2;
            const int64_t i3_0 = i3 / r3;

            const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            const char *src1_batch = src1_base + i3   * nb3_1 + i2   * nb2_1;

            const int64_t mb = mb_idx * TILE_M;
            const int64_t nb = nb_idx * TILE_N;
            const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

            for (int64_t kb = kb_start; kb < kb_end; ++kb) {
                int buf = chunk_id & 1;
                if (chunk_id >= 2) scp_wait(consumed_ctr, chunk_id - 1);

                read_act_block_q8_0(a_panel[buf], sa_panel[buf], src1_batch, nb, kb, n_cur, nb1_1);
                unpack_weight_block(b_panel[buf], dw_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2(a_panel[buf], PANEL_BYTES / 64, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }
        FENCE;
        return 0;
    }

    // ================================================================
    // Hart 0: tensor pipeline (IMA8A32 -> tensor_quant -> fadd accumulate)
    // ================================================================
    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    // The accumulator zeroing (fbci.pi) and per-block accumulate (fadd.ps) below
    // are packed VPU ops that execute under the m0 lane mask. Unlike the pure-
    // tensor FP16/FP32 ME kernels, this kernel touches .pi/.ps on hart 0, so we
    // must enable all 8 lanes — otherwise masked-off lanes keep stale values and
    // a fraction of each output column comes out wrong.
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    evict_to_l2((const void *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_to_l2((const void *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;
    for (int64_t tile = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
         tile < base_tiles; tile += tiles_stride) {

        const int64_t tiles_per_batch = m_tiles * n_tiles;
        const int64_t batch_idx       = tile / tiles_per_batch;
        const int64_t tile_in_batch   = tile % tiles_per_batch;
        const int64_t nb_idx = tile_in_batch / m_tiles;
        const int64_t mb_idx = tile_in_batch % m_tiles;

        const int64_t i3 = batch_idx / ne2_1;
        const int64_t i2 = batch_idx % ne2_1;
        char *dst_batch = dst_base + i3 * nb3_d + i2 * nb2_d;

        const int64_t mb = mb_idx * TILE_M;
        const int64_t nb = nb_idx * TILE_N;
        const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Zero the FP32 accumulator registers f16..f(16+2*n_cur-1).
        for (int r = 0; r < n_cur * 2; ++r) {
            int reg = ACC_REG_START + r;
            switch (reg) {
                #define ZR(R) case R: __asm__ __volatile__("fbci.pi f" #R ", 0" ::: "f" #R); break;
                ZR(16) ZR(17) ZR(18) ZR(19) ZR(20) ZR(21) ZR(22) ZR(23)
                ZR(24) ZR(25) ZR(26) ZR(27) ZR(28) ZR(29) ZR(30) ZR(31)
                #undef ZR
                default: break;
            }
        }

        for (int64_t kb = kb_start; kb < kb_end; ++kb) {
            int buf = chunk_id & 1;
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load A int8 (plain) and the scale vectors into L1 SCP. B is fed
            // through TenB (TensorLoadSetupB) rather than a plain L1-SCP load:
            // on real hardware the INT8 TensorIMA8A32 only reads the B operand
            // correctly for all 16 output columns when B lives in TenB (a plain
            // L1-SCP B silently drops the 4th column-group, m=12..15). The TenB
            // load is paired with — and completed by — the TensorFMA below, and
            // requires brows(B) == acols (both 8 here).
            tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) a_panel[buf], 0, n_cur - 1, 64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);
            tensor_load(false, false, S_L1_START, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) dw_panel[buf], 0, 0, 64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);
            tensor_load(false, false, S_L1_START + 1, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) sa_panel[buf], 0, 0, 64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);

            // B (32 x 16 int8, interleaved) -> TenB; pairs with the FMA.
            tensor_load_setup_b(false, (uint64_t) b_panel[buf], 7, 64, 1);

            // TensorIMA8A32: int8 A (n_cur x 32) @ int8 B (32 x 16) -> int32.
            // a_num_cols = (32/4)-1 = 7, b_num_col = (16/4)-1 = 3.
            // tenb_loc=1 (B from TenB), first_pass=1, tenc2rf=1 (-> FREGS f0..).
            tensor_fma(false, 3, n_cur - 1, 7, 0,
                       true,            // tenc_loc -> bit23 tenc2rf: copy to FREGS
                       false, false,
                       true,            // tenb_loc=1: B operand from TenB
                       0, A_L1_START, TENSOR_FMA_OP_INT8, true);
            tensor_wait(TENSOR_FMA_WAIT);

            // tensor_quant reads EVERY vector-consuming transform from the same
            // single scratchpad line (scp_loc), so the per-column d_w and the
            // per-row scale_a cannot share one call — split into two passes,
            // each pointing at its own line.
            // Pass 1: INT32->FP32, then *d_w per output column m (line S = dw).
            tensor_quant(0, 3, n_cur - 1, S_L1_START,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         QUANT_FP32_MUL_ROW,    // transf1: *d_w (per col m)
                         QUANT_INT32_TO_FP32);  // transf0
            tensor_wait(TENSOR_QUANT_WAIT);
            // Pass 2: *scale_a per output row n (line S+1 = sa).
            tensor_quant(0, 3, n_cur - 1, S_L1_START + 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0,
                         QUANT_FP32_MUL_COL);   // transf0: *scale_a (per row n)
            tensor_wait(TENSOR_QUANT_WAIT);

            // Accumulate scaled block (f0..f(2*n_cur-1)) into acc (f16..).
            for (int r = 0; r < n_cur * 2; ++r) {
                int sreg = r;
                int dreg = ACC_REG_START + r;
                switch ((sreg << 8) | dreg) {
                    #define FA(S,D) case ((S<<8)|D): __asm__ __volatile__("fadd.ps f" #D ", f" #D ", f" #S) ; break;
                    FA(0,16) FA(1,17) FA(2,18) FA(3,19) FA(4,20) FA(5,21) FA(6,22) FA(7,23)
                    FA(8,24) FA(9,25) FA(10,26) FA(11,27) FA(12,28) FA(13,29) FA(14,30) FA(15,31)
                    #undef FA
                    default: break;
                }
            }

            scp_signal(consumed_ctr, chunk_id);
        }

        // Store FP32 accumulator (f16..) to dst.
        tensor_store(0, ACC_REG_START, 3, n_cur - 1,
                     (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                     0, (uint64_t) nb1_d);
        tensor_wait(TENSOR_STORE_WAIT);
    }

    FENCE;
    return 0;
}
