//******************************************************************************
// MUL_MAT_ID kernel specialized for Q4_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q4_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy (V4, work-balanced partitioning + streaming pair-merge):
//   - Work unit = (non-empty expert, m-pair). Its cost is ~proportional to
//     expert_cnt[e] (the streaming pair-merge touches every routing of that
//     expert). The old V3 split EXPERTS evenly across shires, so the shire
//     pinned to the hottest expert became a serial bottleneck under skewed
//     MoE routing while the rest idled.
//   - V4 instead gives every hart an equal share of WORK: it splits the unit
//     stream by cumulative weight (sum of expert_cnt over a hart's units),
//     resolved against a per-expert weight prefix. Each hart owns a contiguous
//     [u_start, u_end) range, so a cold expert that fits in one hart-block stays
//     L2-local, while a hot expert automatically fans out across many harts.
//   - Locality bonus: at uniform load every expert weighs the same, so each
//     expert lands on exactly HARTS_PER_SHIRE contiguous harts (one shire) —
//     the same L2 reuse V3 had, but without the load imbalance.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

#define MAX_N_EXPERT 128

// Fallback: original m-major iteration. Used when alignment is unsuitable
// or n_expert exceeds the stack-budget cap.
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

    const int64_t K_blocks       = K / QK4_0;
    const int     use_x2         = ((nb01 & 31) == 0);
    const int64_t total_routings = n_expert_used * batch;

    if (total_routings == 0 || M == 0) return 0;

    // Fall back when alignment is unsuitable or expert count exceeds cap.
    if (!use_x2 || n_expert > MAX_N_EXPERT) {
        q4_dot_state q4_state;
        q4_dot_begin(&q4_state);
        run_simple(thread_id, num_threads, src0, src1, src2, dst);
        q4_dot_end(&q4_state);
        return 0;
    }

    // ---- One scan of src2 → per-expert counts (~512 B stack) ----
    int32_t expert_cnt[MAX_N_EXPERT];
    int32_t nonempty[MAX_N_EXPERT];
    for (int64_t e = 0; e < n_expert; e++) expert_cnt[e] = 0;

    int32_t any_invalid = 0;
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t e = *(const int32_t*)((const char*)src2_data
                              + s * (int64_t)nb20 + b * (int64_t)nb21);
            if (e >= 0 && e < n_expert) expert_cnt[e]++;
            else                        any_invalid = 1;
        }
    }

    int32_t n_nonempty = 0;
    for (int64_t e = 0; e < n_expert; e++) {
        if (expert_cnt[e] > 0) nonempty[n_nonempty++] = (int32_t)e;
    }

    // Hart 0 zeros outputs of invalid routings (cheap, almost never hit).
    if (any_invalid && thread_id == 0) {
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

    // ---- Work-balanced partitioning ----
    // Unit = (non-empty expert, m-pair); its cost ~= expert_cnt[e] (the inner
    // streaming pair-merge walks every routing of that expert). Give each hart
    // an equal share of total WORK (= sum of expert_cnt over its units) instead
    // of an equal count of experts, so a hot expert spreads across many harts.
    const int64_t m_pairs_per_expert = (M + 1) / 2;
    const int64_t total_units        = (int64_t)n_nonempty * m_pairs_per_expert;

    // Total work weight W = sum_e expert_cnt[e] * m_pairs_per_expert.
    int64_t W = 0;
    for (int32_t i = 0; i < n_nonempty; i++) {
        W += (int64_t)expert_cnt[nonempty[i]] * m_pairs_per_expert;
    }
    if (W == 0) return 0;

    // This hart owns the units whose cumulative weight lands in [tw_lo, tw_hi).
    const int64_t tw_lo = ((int64_t)thread_id       * W) / (int64_t)num_threads;
    const int64_t tw_hi = ((int64_t)(thread_id + 1) * W) / (int64_t)num_threads;

    // Resolve each weight boundary to a unit-aligned global index
    // u = nei * m_pairs_per_expert + mp, walking the per-expert weight prefix.
    int64_t u_start = total_units;
    int64_t u_end   = total_units;
    {
        int64_t acc = 0;
        for (int32_t i = 0; i < n_nonempty; i++) {
            const int64_t wpm  = (int64_t)expert_cnt[nonempty[i]]; // weight per m-pair
            const int64_t wexp = wpm * m_pairs_per_expert;         // weight of whole expert
            if (u_start == total_units && acc + wexp > tw_lo) {
                u_start = (int64_t)i * m_pairs_per_expert + (tw_lo - acc) / wpm;
            }
            if (u_end == total_units && acc + wexp > tw_hi) {
                u_end = (int64_t)i * m_pairs_per_expert + (tw_hi - acc) / wpm;
                break;
            }
            acc += wexp;
        }
    }
    if (u_start >= u_end) return 0;

    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    for (int64_t u = u_start; u < u_end; u++) {
        const int32_t nei = (int32_t)(u / m_pairs_per_expert);
        const int64_t mp  = u - (int64_t)nei * m_pairs_per_expert;
        const int32_t e   = nonempty[nei];

        const int64_t m0     = 2 * mp;
        const int64_t m1     = m0 + 1;
        const int     has_m1 = (m1 < M);

        const char*       ebase = (const char*)src0_data + (int64_t)e * (int64_t)nb02;
        const block_q4_0* row0  = (const block_q4_0*)(ebase + m0 * (int64_t)nb01);
        const block_q4_0* row1  = has_m1
            ? (const block_q4_0*)(ebase + m1 * (int64_t)nb01)
            : NULL;

        // Streaming pair-merge over src2: on each hit for this expert,
        // either stash as pending or pair with the pending one and emit.
        int32_t  have_pending = 0;
        uint32_t s_p = 0, b_p = 0;

        for (int64_t b = 0; b < batch; b++) {
            for (int64_t s = 0; s < n_expert_used; s++) {
                const int32_t eid = *(const int32_t*)((const char*)src2_data
                                    + s * (int64_t)nb20 + b * (int64_t)nb21);
                if (eid != e) continue;

                if (!have_pending) {
                    s_p = (uint32_t)s;
                    b_p = (uint32_t)b;
                    have_pending = 1;
                    continue;
                }

                // Pair (s_p, b_p) with current (s, b).
                const float* b_col0 = (const float*)((const char*)src1_data
                    + ((int64_t)s_p % ne11) * (int64_t)nb11
                    + (int64_t)b_p * (int64_t)nb12);
                const float* b_col1 = (const float*)((const char*)src1_data
                    + (s % ne11) * (int64_t)nb11
                    + b * (int64_t)nb12);

                char* d0 = (char*)dst_data + (int64_t)s_p * (int64_t)nbd1 + (int64_t)b_p * (int64_t)nbd2;
                char* d1 = (char*)dst_data + s * (int64_t)nbd1 + b * (int64_t)nbd2;

                if (has_m1) {
                    float r00, r01, r10, r11;
                    q4_dot_compute_2x2_aligned(row0, row1, b_col0, b_col1,
                                               K_blocks, &r00, &r01, &r10, &r11);
                    atomic_store_f32((volatile float*)(d0 + m0 * (int64_t)nbd0), r00);
                    atomic_store_f32((volatile float*)(d1 + m0 * (int64_t)nbd0), r01);
                    atomic_store_f32((volatile float*)(d0 + m1 * (int64_t)nbd0), r10);
                    atomic_store_f32((volatile float*)(d1 + m1 * (int64_t)nbd0), r11);
                } else {
                    // m1 out of bounds — only m0 row exists; do two single dots.
                    float sa = q4_dot_compute(row0, b_col0, K_blocks);
                    float sb = q4_dot_compute(row0, b_col1, K_blocks);
                    atomic_store_f32((volatile float*)(d0 + m0 * (int64_t)nbd0), sa);
                    atomic_store_f32((volatile float*)(d1 + m0 * (int64_t)nbd0), sb);
                }
                have_pending = 0;
            }
        }

        if (have_pending) {
            const float* b_col = (const float*)((const char*)src1_data
                + ((int64_t)s_p % ne11) * (int64_t)nb11
                + (int64_t)b_p * (int64_t)nb12);
            char* ds = (char*)dst_data + (int64_t)s_p * (int64_t)nbd1 + (int64_t)b_p * (int64_t)nbd2;
            if (has_m1) {
                float s0v, s1v;
                q4_dot_compute_x2_aligned(row0, row1, b_col, K_blocks, &s0v, &s1v);
                atomic_store_f32((volatile float*)(ds + m0 * (int64_t)nbd0), s0v);
                atomic_store_f32((volatile float*)(ds + m1 * (int64_t)nbd0), s1v);
            } else {
                float sum = q4_dot_compute(row0, b_col, K_blocks);
                atomic_store_f32((volatile float*)(ds + m0 * (int64_t)nbd0), sum);
            }
        }
    }

    q4_dot_end(&q4_state);
    return 0;
}
