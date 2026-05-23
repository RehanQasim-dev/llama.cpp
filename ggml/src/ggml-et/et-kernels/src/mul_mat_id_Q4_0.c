//******************************************************************************
// MUL_MAT_ID kernel specialized for Q4_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q4_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy (prefill-friendly 2x2 tile):
//   - Bucket every (slot, batch) routing by expert id (counting sort, done
//     locally on each hart). Outputs that share an expert can then be paired
//     across activation rows, not only across `m`.
//   - Work unit = (non-empty expert, m-pair). Harts each own a contiguous
//     chunk so harts with light routes don't sit idle.
//   - Inside a unit we walk that expert's slot list in pairs and call the
//     2x2 dot: two weight rows x two B columns -> four outputs, reusing
//     both A loads across columns and B loads across rows.
//   - Tail cases: odd slot count -> x2 (2 m, 1 slot); odd m -> single-row
//     dots over each slot. Decode (1 slot/expert) always falls through to x2
//     or single.
//   - Invalid expert ids are zeroed by hart 0 in a direct scan of src2.
//
// Stack budget (per hart):
//   slot_tokens[MAX_SLOT_BUF]  = 256 * 4 =  1 KB
//   expert_off[MAX_N_EXPERT+1] = 129 * 4 =  ~0.5 KB
//   nonempty[MAX_N_EXPERT]     = 128 * 4 =  ~0.5 KB
//   cursor[MAX_N_EXPERT]       = 128 * 4 =  ~0.5 KB
//   Total                                ~= 2.5 KB  (fits easily)
//
// Fallback to the simple m-major loop when:
//   - nb01 is not 32-byte aligned (x2/2x2 asm requires it)
//   - total_routings > MAX_SLOT_BUF (too many to sort on-stack)
//   - n_expert > MAX_N_EXPERT
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

#define MAX_SLOT_BUF  256
#define MAX_N_EXPERT  128

// Pack (slot, batch) into one 32-bit token.
// Both slot < n_expert_used and batch are small in practice (<= 64K).
static inline uint32_t pack_sb(uint32_t slot, uint32_t b) {
    return (b << 16) | (slot & 0xFFFFu);
}
static inline uint32_t unpack_slot(uint32_t tok)  { return tok & 0xFFFFu; }
static inline uint32_t unpack_batch(uint32_t tok) { return tok >> 16; }

// Fallback: original m-major iteration. Used when grouping caps are exceeded
// or when alignment is not suitable for x2/2x2.
static void run_simple(int thread_id, int num_threads,
                       const struct ggml_tensor* src0,
                       const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2,
                       struct ggml_tensor* dst) {
    const void*    src0_data = src0->data;
    const float*   src1_data = (const float*)src1->data;
    const int32_t* src2_data = (const int32_t*)src2->data;
    float*         dst_data  = (float*)dst->data;

    const int64_t K             = src0->ne[0];
    const int64_t M             = src0->ne[1];
    const int64_t n_expert      = src0->ne[2];
    const int64_t n_expert_used = src2->ne[0];
    const int64_t batch         = src2->ne[1];
    const int64_t ne11          = src1->ne[1];
    const size_t  nb01 = src0->nb[1];
    const size_t  nb02 = src0->nb[2];
    const size_t  nb11 = src1->nb[1];
    const size_t  nb12 = src1->nb[2];
    const size_t  nb20 = src2->nb[0];
    const size_t  nb21 = src2->nb[1];
    const size_t  nbd0 = dst->nb[0];
    const size_t  nbd1 = dst->nb[1];
    const size_t  nbd2 = dst->nb[2];
    const int64_t K_blocks = K / QK4_0;
    const int     use_x2   = ((nb01 & 31) == 0);

    const uint64_t total_outputs = (uint64_t)M * (uint64_t)n_expert_used * (uint64_t)batch;
    const uint64_t chunk    = (total_outputs + (uint64_t)num_threads - 1) / (uint64_t)num_threads;
    const uint64_t my_start = (uint64_t)thread_id * chunk;
    if (my_start >= total_outputs) return;
    uint64_t my_end = my_start + chunk;
    if (my_end > total_outputs) my_end = total_outputs;

    const uint64_t per_batch = (uint64_t)M * (uint64_t)n_expert_used;
    uint64_t idx = my_start;
    while (idx < my_end) {
        const int64_t  bi      = (int64_t)(idx / per_batch);
        const uint64_t rem     = idx - (uint64_t)bi * per_batch;
        const int64_t  si      = (int64_t)(rem / (uint64_t)M);
        const int64_t  m0      = (int64_t)(rem - (uint64_t)si * (uint64_t)M);
        const uint64_t run_end = (uint64_t)bi * per_batch + (uint64_t)si * (uint64_t)M + (uint64_t)M;
        const uint64_t end_in  = (run_end < my_end) ? run_end : my_end;
        int64_t run_len = (int64_t)(end_in - idx);

        const int32_t e = *(const int32_t*)((const char*)src2_data
                           + si * (int64_t)nb20 + bi * (int64_t)nb21);
        char* dst_slot = (char*)dst_data + si * (int64_t)nbd1 + bi * (int64_t)nbd2;

        if (e < 0 || e >= n_expert) {
            for (int64_t i = 0; i < run_len; i++)
                atomic_store_f32((volatile float*)(dst_slot + (m0 + i) * (int64_t)nbd0), 0.0f);
            idx += (uint64_t)run_len;
            continue;
        }
        const int64_t  col   = si % ne11;
        const float*   b_col = (const float*)((const char*)src1_data
                                + col * (int64_t)nb11 + bi * (int64_t)nb12);
        const char*    ebase = (const char*)src0_data + (int64_t)e * (int64_t)nb02;
        int64_t m = m0, left = run_len;
        if (use_x2) {
            while (left >= 2) {
                float s0, s1;
                q4_dot_compute_x2_aligned(
                    (const block_q4_0*)(ebase + m       * (int64_t)nb01),
                    (const block_q4_0*)(ebase + (m + 1) * (int64_t)nb01),
                    b_col, K_blocks, &s0, &s1);
                atomic_store_f32((volatile float*)(dst_slot + m       * (int64_t)nbd0), s0);
                atomic_store_f32((volatile float*)(dst_slot + (m + 1) * (int64_t)nbd0), s1);
                m += 2; left -= 2;
            }
        }
        while (left > 0) {
            float s = q4_dot_compute((const block_q4_0*)(ebase + m * (int64_t)nb01), b_col, K_blocks);
            atomic_store_f32((volatile float*)(dst_slot + m * (int64_t)nbd0), s);
            m++; left--;
        }
        idx += (uint64_t)run_len;
    }
}

int entry_point(struct ggml_et_mul_mat_id_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env || !params) return -1;

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* src2 = &params->src2;
    struct ggml_tensor* dst  = &params->dst;

    if (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 ||
        src2->type != GGML_TYPE_I32  || dst->type  != GGML_TYPE_F32)
        return -1;

    const void*    src0_data = src0->data;
    const float*   src1_data = (const float*)src1->data;
    const int32_t* src2_data = (const int32_t*)src2->data;
    float*         dst_data  = (float*)dst->data;
    if (!src0_data || !src1_data || !src2_data || !dst_data) return -1;

    const int64_t K             = src0->ne[0];
    const int64_t M             = src0->ne[1];
    const int64_t n_expert      = src0->ne[2];
    const int64_t n_expert_used = src2->ne[0];
    const int64_t batch         = src2->ne[1];
    const int64_t ne11          = src1->ne[1];

    if (K % QK4_0 != 0) return -1;

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb20 = src2->nb[0];
    const size_t nb21 = src2->nb[1];
    const size_t nbd0 = dst->nb[0];
    const size_t nbd1 = dst->nb[1];
    const size_t nbd2 = dst->nb[2];

    if (src0->nb[0] != sizeof(block_q4_0) || src1->nb[0] != sizeof(float) ||
        src2->nb[0] != sizeof(int32_t)    || nbd0 != sizeof(float))
        return -1;

    const int64_t K_blocks        = K / QK4_0;
    const int     use_x2          = ((nb01 & 31) == 0);
    const int64_t total_routings  = n_expert_used * batch;

    if (total_routings == 0 || M == 0) return 0;

    // Fall back to m-major when caps exceeded or alignment unsuitable.
    if (!use_x2 ||
        total_routings > MAX_SLOT_BUF ||
        n_expert > MAX_N_EXPERT) {
        q4_dot_state q4_state;
        q4_dot_begin(&q4_state);
        run_simple(thread_id, num_threads, src0, src1, src2, dst);
        q4_dot_end(&q4_state);
        return 0;
    }

    // ---- Per-hart counting sort: bucket routings by expert id ----
    // Stack budget: ~2.5 KB total (all arrays sized to caps above).
    int32_t  expert_off[MAX_N_EXPERT + 1];
    uint32_t slot_tokens[MAX_SLOT_BUF];
    int32_t  cursor[MAX_N_EXPERT];
    int32_t  nonempty[MAX_N_EXPERT];

    for (int64_t e = 0; e <= n_expert; e++) expert_off[e] = 0;

    // Count valid routings per expert. Invalid expert ids are handled
    // separately by hart 0 below.
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t e = *(const int32_t*)((const char*)src2_data
                              + s * (int64_t)nb20 + b * (int64_t)nb21);
            if (e >= 0 && e < n_expert) expert_off[e + 1]++;
        }
    }
    // Prefix sum.
    for (int64_t e = 1; e <= n_expert; e++) expert_off[e] += expert_off[e - 1];

    // Place tokens. cursor starts at each expert's base offset.
    for (int64_t e = 0; e < n_expert; e++) cursor[e] = expert_off[e];
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t e = *(const int32_t*)((const char*)src2_data
                              + s * (int64_t)nb20 + b * (int64_t)nb21);
            if (e >= 0 && e < n_expert)
                slot_tokens[cursor[e]++] = pack_sb((uint32_t)s, (uint32_t)b);
        }
    }

    // Build compact list of non-empty experts.
    int32_t n_nonempty = 0;
    for (int64_t e = 0; e < n_expert; e++) {
        if (expert_off[e + 1] > expert_off[e])
            nonempty[n_nonempty++] = (int32_t)e;
    }

    // Hart 0 handles invalid-expert outputs (rare in practice, O(M) per bad
    // routing). No extra storage needed — just scan src2 directly.
    if (thread_id == 0) {
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t s = 0; s < n_expert_used; s++) {
                const int32_t e = *(const int32_t*)((const char*)src2_data
                                  + s * (int64_t)nb20 + b * (int64_t)nb21);
                if (e < 0 || e >= n_expert) {
                    char* dst_slot = (char*)dst_data
                                   + s * (int64_t)nbd1
                                   + b * (int64_t)nbd2;
                    for (int64_t m = 0; m < M; m++)
                        atomic_store_f32(
                            (volatile float*)(dst_slot + m * (int64_t)nbd0), 0.0f);
                }
            }
        }
    }

    if (n_nonempty == 0) return 0;

    // ---- Distribute (non-empty expert, m-pair) work units across harts ----
    const int64_t  m_pairs_per_expert = (M + 1) / 2;
    const uint64_t total_units = (uint64_t)n_nonempty * (uint64_t)m_pairs_per_expert;
    const uint64_t chunk    = (total_units + (uint64_t)num_threads - 1) / (uint64_t)num_threads;
    const uint64_t my_start = (uint64_t)thread_id * chunk;
    if (my_start >= total_units) return 0;
    uint64_t my_end = my_start + chunk;
    if (my_end > total_units) my_end = total_units;

    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    for (uint64_t u = my_start; u < my_end; u++) {
        const int32_t nei   = (int32_t)(u / (uint64_t)m_pairs_per_expert);
        const int64_t mp    = (int64_t)(u - (uint64_t)nei * (uint64_t)m_pairs_per_expert);
        const int32_t e     = nonempty[nei];

        const int64_t m0    = 2 * mp;
        const int64_t m1    = m0 + 1;
        const int     has_m1 = (m1 < M);

        const int32_t slot_lo = expert_off[e];
        const int32_t slot_hi = expert_off[e + 1];

        const char*       ebase = (const char*)src0_data + (int64_t)e * (int64_t)nb02;
        const block_q4_0* row0  = (const block_q4_0*)(ebase + m0 * (int64_t)nb01);
        const block_q4_0* row1  = has_m1
            ? (const block_q4_0*)(ebase + m1 * (int64_t)nb01)
            : NULL;

        int32_t i = slot_lo;

        if (has_m1) {
            // Main path: 2x2 tile — 2 weight rows x 2 B columns -> 4 outputs.
            while (i + 1 < slot_hi) {
                const uint32_t tok0 = slot_tokens[i];
                const uint32_t tok1 = slot_tokens[i + 1];
                const uint32_t s0 = unpack_slot(tok0), b0 = unpack_batch(tok0);
                const uint32_t s1 = unpack_slot(tok1), b1 = unpack_batch(tok1);

                const float* b_col0 = (const float*)((const char*)src1_data
                    + ((int64_t)s0 % ne11) * (int64_t)nb11
                    + (int64_t)b0 * (int64_t)nb12);
                const float* b_col1 = (const float*)((const char*)src1_data
                    + ((int64_t)s1 % ne11) * (int64_t)nb11
                    + (int64_t)b1 * (int64_t)nb12);

                float r00, r01, r10, r11;
                q4_dot_compute_2x2_aligned(row0, row1, b_col0, b_col1,
                                           K_blocks, &r00, &r01, &r10, &r11);

                char* d0 = (char*)dst_data + (int64_t)s0 * (int64_t)nbd1 + (int64_t)b0 * (int64_t)nbd2;
                char* d1 = (char*)dst_data + (int64_t)s1 * (int64_t)nbd1 + (int64_t)b1 * (int64_t)nbd2;
                atomic_store_f32((volatile float*)(d0 + m0 * (int64_t)nbd0), r00);
                atomic_store_f32((volatile float*)(d1 + m0 * (int64_t)nbd0), r01);
                atomic_store_f32((volatile float*)(d0 + m1 * (int64_t)nbd0), r10);
                atomic_store_f32((volatile float*)(d1 + m1 * (int64_t)nbd0), r11);
                i += 2;
            }
            // Odd-slot tail: x2 (2 weight rows x 1 B col).
            if (i < slot_hi) {
                const uint32_t tok = slot_tokens[i];
                const uint32_t s = unpack_slot(tok), b = unpack_batch(tok);
                const float* b_col = (const float*)((const char*)src1_data
                    + ((int64_t)s % ne11) * (int64_t)nb11
                    + (int64_t)b * (int64_t)nb12);
                float s0v, s1v;
                q4_dot_compute_x2_aligned(row0, row1, b_col, K_blocks, &s0v, &s1v);
                char* ds = (char*)dst_data + (int64_t)s * (int64_t)nbd1 + (int64_t)b * (int64_t)nbd2;
                atomic_store_f32((volatile float*)(ds + m0 * (int64_t)nbd0), s0v);
                atomic_store_f32((volatile float*)(ds + m1 * (int64_t)nbd0), s1v);
            }
        } else {
            // Odd-m tail (m1 >= M): single-row dot over each slot.
            // Also the decode path naturally lands here when M is odd.
            for (; i < slot_hi; i++) {
                const uint32_t tok = slot_tokens[i];
                const uint32_t s = unpack_slot(tok), b = unpack_batch(tok);
                const float* b_col = (const float*)((const char*)src1_data
                    + ((int64_t)s % ne11) * (int64_t)nb11
                    + (int64_t)b * (int64_t)nb12);
                float sum = q4_dot_compute(row0, b_col, K_blocks);
                char* ds = (char*)dst_data + (int64_t)s * (int64_t)nbd1 + (int64_t)b * (int64_t)nbd2;
                atomic_store_f32((volatile float*)(ds + m0 * (int64_t)nbd0), sum);
            }
        }
    }

    q4_dot_end(&q4_state);
    return 0;
}
