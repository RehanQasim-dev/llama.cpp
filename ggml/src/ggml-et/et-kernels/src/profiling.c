#include <trace/trace_umode.h>
#include <etsoc/common/utils.h>

void et_profile_start() {
    int hid = get_hart_id();

    // Compute PMCs (Instructions/L2 misses) have 2 shared slots per neighborhood (even/odd).
    // Hart 0 reads the sum of all even harts, Hart 1 reads the sum of all odd harts.
    // Only 2 harts per neighborhood need to log this.
    if (hid % 16 == 0 || hid % 16 == 1) {
        et_trace_pmc_compute(hid);
    }

    // SC PMCs (Shire Caches) are per-neighborhood. Only 1 thread per neighborhood should log them.
    if (hid % 16 == 0) {
        et_trace_pmc_sc(hid);
    }

    // MS PMCs (Memory Shires) are system-wide (8 total). Only 1 thread overall should log them.
    if (hid == 0) {
        for (int i = 0; i < 8; i++) {
            et_trace_pmc_ms(hid, i);
        }
    }
}

void et_profile_end() {
    int hid = get_hart_id();

    if (hid % 16 == 0 || hid % 16 == 1) {
        et_trace_pmc_compute(hid);
    }

    if (hid % 16 == 0) {
        et_trace_pmc_sc(hid);
    }

    if (hid == 0) {
        for (int i = 0; i < 8; i++) {
            et_trace_pmc_ms(hid, i);
        }
    }
}
