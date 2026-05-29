//******************************************************************************
// QUANTIZE_MAT F32 -> Q8_0
// Quantizes an F32 activation matrix [K, N(...)] to Q8_0 blocks, matching
// ggml's quantize_row_q8_0_ref (d = amax/127; qs = round(x/d); d stored fp16).
// Used as a pre-pass before the int8 Q4_0 mul_mat kernel.
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"

struct ggml_et_cont_params {
    struct ggml_tensor src0;   // F32 input  [K, N, ...]
    struct ggml_tensor dst;    // Q8_0 output
};

int entry_point(struct ggml_et_cont_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params) return -1;

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    const struct ggml_tensor * src0 = &params->src0;
    const struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_Q8_0) return -1;

    const char * src_data = (const char *) src0->data;
    char       * dst_data = (char *) dst->data;
    if (!src_data || !dst_data) return -1;

    const int64_t K   = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t nbd1 = dst->nb[1];
    const int64_t nbd2 = dst->nb[2];
    const int64_t nbd3 = dst->nb[3];

    if (K % QK8_0 != 0) return -1;
    const int64_t n_blocks = K / QK8_0;

    // Distribute whole rows (one per (i1,i2,i3)) across threads.
    const int64_t total_rows = ne1 * ne2 * ne3;
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t r_start = (int64_t) thread_id * rows_per_thread;
    int64_t r_end = r_start + rows_per_thread;
    if (r_end > total_rows) r_end = total_rows;
    if (r_start >= total_rows) return 0;

    for (int64_t r = r_start; r < r_end; ++r) {
        const int64_t i1 = r % ne1;
        const int64_t i2 = (r / ne1) % ne2;
        const int64_t i3 = r / (ne1 * ne2);

        const char * src_row = src_data + i1 * nb01 + i2 * nb02 + i3 * nb03;
        char       * dst_row = dst_data + i1 * nbd1 + i2 * nbd2 + i3 * nbd3;

        for (int64_t kb = 0; kb < n_blocks; ++kb) {
            const char * xb = src_row + kb * QK8_0 * nb00;

            float amax = 0.0f;
            for (int j = 0; j < QK8_0; ++j) {
                const float v = *(const float *)(xb + j * nb00);
                const float av = v < 0.0f ? -v : v;
                if (av > amax) amax = av;
            }

            const float d  = amax * (1.0f / 127.0f);
            const float id = d > 0.0f ? et_fdiv(1.0f, d) : 0.0f;

            block_q8_0 * blk = (block_q8_0 *)(dst_row + kb * (int64_t) sizeof(block_q8_0));
            blk->d = fp32_to_fp16(d);

            for (int j = 0; j < QK8_0; ++j) {
                const float x = *(const float *)(xb + j * nb00) * id;
                int q = (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
                blk->qs[j] = (int8_t) q;
            }
        }
    }

    return 0;
}
