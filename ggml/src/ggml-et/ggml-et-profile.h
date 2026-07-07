#pragma once

#include <cstddef>
#include <string>

// -----------------------------------------------------------------------------
// Host-side ET-SOC1 performance-counter profiling.
//
// This mirrors the standalone et-kernel-bench launcher: while profiling is
// armed, every kernel launch runs with user tracing enabled, the device-side
// profile.c wrapper emits PMC snapshots before/after the kernel, and the host
// decodes the trace buffer into per-shire / per-memory-shire deltas. Deltas are
// accumulated per kernel name and averaged when profiling ends.
//
// Used by test-backend-ops perf mode via the backend-reg proc-address hooks
// ggml_backend_et_perf_counters_begin / _end. All calls happen on the single
// host thread that drives the default stream, so no locking is needed.
// -----------------------------------------------------------------------------

// True while a begin()/end() window is open and counters should be collected.
bool ggml_et_profile_active();

// Clear accumulators and arm collection.
void ggml_et_profile_begin();

// Average the accumulated counters, print a table labelled with `label`
// (typically the test-case description), and disarm collection.
void ggml_et_profile_end(const char * label);

// Decode one kernel launch's trace buffer and add its deltas to the running
// total for `kernel_name`. No-op if profiling is not active.
void ggml_et_profile_accumulate(const std::string & kernel_name,
                                const void * trace_buf, size_t trace_size);
