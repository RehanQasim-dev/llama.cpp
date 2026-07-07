//******************************************************************************
// profile.c — PMC-snapshot wrapper around a kernel's entry point.
//
// Each standalone kernel is compiled with -Dentry_point=user_kernel, so the
// kernel's own entry becomes user_kernel(). This file provides et_kernel_entry()
// (the symbol crt.S calls): it snapshots the hardware performance counters
// immediately before and after user_kernel(), leaving two trace packets per
// active hardware unit for the host to decode into a delta.
//
// et_kernel_entry is a fixed symbol that is deliberately NOT affected by the
// -Dentry_point=user_kernel rename, so this wrapper and the kernel body never
// collide on a name and crt.S always lands here first.
//
// The et_trace_pmc_* macros are self-gating: when the host does not enable
// user tracing, firmware sets the per-hart trace control block to
// TRACE_DISABLE and every snapshot below is a single guarded no-op. So this
// wrapper is inert (and safe) on the default, non-profiled launch path.
//
// Leader-thread selection (matches the host decoder in ggml-et-profile.cpp):
//   compute (hpm3-hpm8) : hart_id % 16 == 0 or 1  (thread 0/1 of each neighborhood)
//   shire-cache (L2)    : hart_id % 16 == 0
//   mem-shire (DRAM)    : global hart 0 loops over all 8 memory shires
//******************************************************************************

#include <trace/trace_umode.h>
#include <etsoc/isa/hart.h>

/* The real kernel, renamed via -Dentry_point=user_kernel at compile time.
 * Declared with an opaque signature — the RISC-V ABI passes params in a0 and
 * env in a1 regardless of the kernel's concrete parameter struct type. */
extern int user_kernel(void *params, void *env);

int et_kernel_entry(void *params, void *env) {
    uint32_t hid = get_hart_id();

    /* --- pre-kernel snapshot --- */
    if (hid % 16 == 0 || hid % 16 == 1) et_trace_pmc_compute(hid);
    if (hid % 16 == 0)                  et_trace_pmc_sc(hid);
    if (hid == 0) {
        for (uint8_t ms = 0; ms < 8; ms++) et_trace_pmc_ms(0, ms);
    }

    int rc = user_kernel(params, env);

    /* --- post-kernel snapshot --- */
    if (hid % 16 == 0 || hid % 16 == 1) et_trace_pmc_compute(hid);
    if (hid % 16 == 0)                  et_trace_pmc_sc(hid);
    if (hid == 0) {
        for (uint8_t ms = 0; ms < 8; ms++) et_trace_pmc_ms(0, ms);
    }

    return rc;
}
