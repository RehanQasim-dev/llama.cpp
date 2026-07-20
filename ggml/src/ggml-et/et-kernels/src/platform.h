//******************************************************************************
// ET Platform Hardware Abstraction Layer
// Provides thread coordination, kernel infrastructure, and platform primitives
// for bare metal ET kernels
//******************************************************************************

#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include "etsoc/isa/hart.h"
#include "etsoc/common/utils.h"
#include "etsoc/isa/barriers.h"  // shire_barrier(), used by et_barrier() below

#define SOC_MINIONS_PER_SHIRE 32
#define NUM_HARTS_PER_MINION 2

// Environment structure definition
typedef struct {
    uint32_t version;           // Version of the ABI (offset 0)
    uint32_t padding1;          // Padding to align shire_mask to offset 8
    uint64_t shire_mask;        // Bitmask of active compute shires (offset 8)
    uint32_t frequency;         // Frequency of Minion cores in MHz (offset 16)
    uint32_t padding2;          // Padding to maintain alignment
} __attribute__((packed, aligned(64))) kernel_environment_t;

// Manual implementation of count trailing zeros for bare metal environment
// NOTE: This simple loop-based implementation is used for portability.
// Production implementations (like libgcc's __ctzdi2) use optimized bit manipulation
// algorithms with lookup tables and parallel bit operations for O(log n) performance.
static inline int manual_ctzll(uint64_t x) {
    if (x == 0) return 64;
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
}

// Manual implementation of population count for bare metal environment
// NOTE: This simple loop-based implementation is used for portability.
// Production implementations (like libgcc's __popcountdi2) use optimized bit-parallel
// algorithms with magic constants and bit manipulation tricks for O(1) performance.
static inline int manual_popcountll(uint64_t x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

// Calculate relative thread ID from absolute hart ID using shire mask
// Returns -1 if this hart is not active (not in shire mask)
static inline int get_relative_thread_id(uint64_t shire_mask) {
    int hart_id = (int)get_hart_id();

    // Find starting hart offset from lowest active shire
    int starting_hart = manual_ctzll(shire_mask) * SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;

    // Return -1 if not an active thread
    if (hart_id < starting_hart) {
        return -1;
    }

    // Calculate relative thread ID
    int thread_id = hart_id - starting_hart;
    return thread_id;
}

// Calculate total number of threads from shire mask
static inline int get_num_threads(uint64_t shire_mask) {
    // Count active shires using popcount, multiply by minions per shire and harts per minion
    return manual_popcountll(shire_mask) * SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;
}

//******************************************************************************
// Synchronization Primitives
//******************************************************************************

#define NOP   __asm__ __volatile__ ("nop\n");
// "memory" clobber added (vs. bare "fence\n") so the barrier/SCP-signaling
// primitives below (which rely on FENCE as a compiler barrier, not just a
// hardware one) can't have their surrounding loads/stores reordered across it.
#define FENCE __asm__ __volatile__ ("fence\n" ::: "memory");
#define WFI   __asm__ __volatile__ ("wfi\n");

//******************************************************************************
// Tensor Engine Wait & Error Macros
//
// These write to CSR 0x830 (tensor_wait) to stall the hart until the specified
// tensor unit completes its current operation.  The immediate encodes which
// unit to wait on.
//******************************************************************************

#define WAIT_TENSOR_LOAD_0     __asm__ __volatile__ ( "csrwi 0x830, 0\n"  : : );
#define WAIT_TENSOR_LOAD_1     __asm__ __volatile__ ( "csrwi 0x830, 1\n"  : : );
#define WAIT_TENSOR_LOAD_L2_0  __asm__ __volatile__ ( "csrwi 0x830, 2\n"  : : );
#define WAIT_TENSOR_LOAD_L2_1  __asm__ __volatile__ ( "csrwi 0x830, 3\n"  : : );
#define WAIT_PREFETCH_0        __asm__ __volatile__ ( "csrwi 0x830, 4\n"  : : );
#define WAIT_PREFETCH_1        __asm__ __volatile__ ( "csrwi 0x830, 5\n"  : : );
#define WAIT_CACHEOPS          __asm__ __volatile__ ( "csrwi 0x830, 6\n"  : : );
#define WAIT_TENSOR_FMA        __asm__ __volatile__ ( "csrwi 0x830, 7\n"  : : );
#define WAIT_TENSOR_STORE      __asm__ __volatile__ ( "csrwi 0x830, 8\n"  : : );
#define WAIT_TENSOR_REDUCE     __asm__ __volatile__ ( "csrwi 0x830, 9\n"  : : );
#define WAIT_TENSOR_QUANT      __asm__ __volatile__ ( "csrwi 0x830, 10\n" : : );
#define STALL                  __asm__ __volatile__ ( "csrw stall, x0\n"  : : );

// Write 0 to CSR 0x808 (tensor_error) to clear any latched tensor error bits.
// Must be issued before the first tensor operation in a kernel to avoid stale
// errors from a previous invocation causing spurious faults.
#define CLEAR_TENSOR_ERROR     __asm__ __volatile__ ( "csrwi 0x808, 0" : : );

//******************************************************************************
// L1 Data Cache / Scratchpad (SCP) Configuration
//
// The ET-SoC-1 L1 data cache can be split so that half its ways operate as a
// software-managed scratchpad (SCP).  Tensor load/store/FMA instructions
// require SCP mode to be active.
//
// CSR 0x810 — ucache_control:
//
//   Bit(s)  Field         Description
//   ──────  ────────────  ──────────────────────────────────────────────────
//   [0]     D1Split       1 = L1 is split (half cache, half SCP).
//                          Read-only from U-mode; set by M-mode firmware
//                          before kernel launch.  Writing ScpEnable while
//                          D1Split=0 is silently ignored.
//   [1]     ScpEnable     1 = scratchpad is active and zeroed.
//   [4:2]   RepRate       Cache-op replay rate (0 = no delay between ops).
//   [10:6]  CacheOpMax    Max outstanding cache ops (0 = unlimited).
//
// Typical kernel prologue for tensor operations:
//     setup_cache_scp();   // enables SCP, waits for zeroing
//     CLEAR_TENSOR_ERROR;  // clear stale error bits
//******************************************************************************

// Write the ucache_control CSR (0x810).
//
//   scp_en       — 1 to enable SCP mode (requires D1Split already set)
//   cacheop_rate — cache-op replay rate (0–7; 0 = no delay)
//   cacheop_max  — max outstanding cache ops (0–31; 0 = unlimited)
static inline void __attribute__((always_inline))
ucache_control(uint64_t scp_en, uint64_t cacheop_rate, uint64_t cacheop_max)
{
    uint64_t csr_enc = ((cacheop_max & 0x1F) << 6) |
                       ((cacheop_rate & 0x7)  << 2) |
                       ((scp_en & 0x1)        << 1);

    __asm__ __volatile__("csrw 0x810, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) : "x31");
}

// Enable L1 scratchpad mode and wait for the transition to complete.
// After this call the SCP lines are zeroed and ready for tensor operations.
//
// Prerequisites:
//   - D1Split must already be 1 (set by M-mode firmware at boot).
//   - Only even harts (hart 0 per minion) should call this, as only they
//     can issue tensor instructions.
static inline void setup_cache_scp(void)
{
    FENCE;                    // drain pending stores before reconfiguring cache
    ucache_control(1, 0, 0);  // ScpEnable=1
    WAIT_CACHEOPS;            // wait for SCP mode transition + zeroing
}

//******************************************************************************
// Atomic Operations
//******************************************************************************

// Atomic store for F32 values to global memory
// Uses ET hardware's custom amoswapg.w instruction for global atomic swap
// This ensures cache coherency when multiple threads write to nearby addresses
static inline void atomic_store_f32(volatile float* addr, float value) {
    uint32_t value_bits = *(uint32_t*)&value;
    __asm__ volatile(
        "amoswapg.w zero, %1, (%0)"
        :
        : "r"(addr), "r"(value_bits)
        : "memory"
    );
}

// Atomic add for F32 values to global memory
// Uses ET hardware's custom amoaddg.w instruction for global atomic add
// This ensures correct accumulation when multiple threads contribute to the same output
static inline void atomic_add_f32(volatile float* addr, float value) {
    uint32_t value_bits = *(uint32_t*)&value;
    __asm__ volatile(
        "amoaddg.w zero, %1, (%0)"
        :
        : "r"(addr), "r"(value_bits)
        : "memory"
    );
}

// Atomic store for F16 values to global memory
// Uses ET hardware's custom shg instruction (store halfword global)
// This ensures cache coherency when multiple threads write to nearby addresses
// Address must be 16-bit aligned
static inline void atomic_store_f16(volatile uint16_t* addr, uint16_t value) {
    __asm__ volatile(
        "shg %1, (%0)"
        :
        : "r"(addr), "r"(value)
        : "memory"
    );
}

//******************************************************************************
// Barrier & L2-SCP-signaling primitives, needed by mul_mat_Q8_0_matrix_engine.c's
// cross-hart K-split reuse path and mul_mat_Q8_0.c's dual-hart K-split. Only
// ET_BARRIER_MINION is used (no SHIRE/GLOBAL scope, no global-AMO barrier).
//******************************************************************************

typedef enum {
    ET_BARRIER_MINION,  // sync both harts within each minion (FLB=minion_id, FCC 0)
} et_barrier_scope_t;

// Barrier with scope-derived parameters. Returns 1 if this hart was last to arrive.
static inline uint64_t __attribute__((always_inline)) et_barrier(et_barrier_scope_t scope) {
    (void) scope;  // only ET_BARRIER_MINION exists/is passed
    uint32_t local_minion = (get_hart_id() >> 1) & 0x1F;
    uint32_t mask         = 1u << local_minion;
    return shire_barrier(local_minion, 0, 2, mask, mask);
}

// One-way semaphore between harts (non-blocking post, blocking wait).
//
// et_sem_post(): increment the partner hart's semaphore. Non-blocking, the
// caller continues immediately. Multiple posts accumulate.
// et_sem_wait(): block until the semaphore is non-zero, then decrement it.
//
// Backed by hardware FCC (Flow Control Credit) counters, FCC 0 for
// ET_BARRIER_MINION scope. Counters are per-hart private, so both harts can
// post/wait on the same scope independently.
//
// Must not be mixed with et_barrier() of the same scope in the same kernel
// (shared FCC channel).
static inline void __attribute__((always_inline)) et_sem_post(et_barrier_scope_t scope) {
    if (scope == ET_BARRIER_MINION) {
        uint64_t hart_id      = get_hart_id();
        uint32_t local_minion = (hart_id >> 1) & 0x1F;
        uint32_t mask         = 1u << local_minion;
        uint64_t shire_id     = get_shire_id();

        if (hart_id & 1) {
            fcc_send(shire_id, THREAD_0, FCC_0, mask);
        } else {
            fcc_send(shire_id, THREAD_1, FCC_0, mask);
        }
    }
}

// Block until a post from et_sem_post() is available, then consume it.
static inline void __attribute__((always_inline)) et_sem_wait(et_barrier_scope_t scope) {
    if (scope == ET_BARRIER_MINION) {
        fcc_consume(FCC_0);
    }
}

//******************************************************************************
// L2 Scratchpad (L2 SCP) Address Computation
//******************************************************************************

#define L2SCP_BASE        0x0080000000ULL
#define L2SCP_SHIRE_LOCAL 0x7FULL

// Format 0: local shire shorthand — no cross-shire traffic.
static inline void * __attribute__((always_inline)) et_shire_l2scp_local(uint64_t offset) {
    return (void *) (L2SCP_BASE | (L2SCP_SHIRE_LOCAL << 23) | (offset & 0x7FFFFF));
}

//******************************************************************************
// Cache Operations (L1 -> L2 flush/evict)
//******************************************************************************

// Flush nlines cache lines at stride apart starting at addr from L1 to L2.
// Uses FlushVA (CSR 0x8BF). Caller must FENCE before and WAIT_CACHEOPS after.
// NOTE: nlines is encoded in a 4-bit field (max 16). DO NOT pass nlines > 16.
static inline void __attribute__((always_inline)) flush_to_l2(const void * addr, uint64_t nlines, uint64_t stride) {
    uint64_t csr_val = (0x1ULL << 58) | ((uint64_t) addr & 0xFFFFFFFFFFC0ULL) | ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x8BF, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory");
}

// Flush an arbitrary number of lines to L2, working around the 16-line cap of a
// single flush_to_l2 by issuing multiple flushes.
static inline void __attribute__((always_inline)) flush_to_l2_multi(const void * addr, uint64_t nlines, uint64_t stride) {
    const char * p = (const char *) addr;
    while (nlines > 16) {
        flush_to_l2(p, 16, stride);
        p += 16 * stride;
        nlines -= 16;
    }
    if (nlines) {
        flush_to_l2(p, nlines, stride);
    }
}

// Evict nlines cache lines at stride apart starting at addr from L1 to L2.
// Uses EvictVA (CSR 0x89F). Guarantees the line is NOT present in L1 after -
// subsequent loads will miss and go to L2/SCP. Caller must FENCE before and
// WAIT_CACHEOPS after. NOTE: nlines max 16 (4-bit field).
static inline void __attribute__((always_inline)) evict_to_l2(const void * addr, uint64_t nlines, uint64_t stride) {
    uint64_t csr_val = (0x1ULL << 58) | ((uint64_t) addr & 0xFFFFFFFFFFC0ULL) | ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x89F, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory");
}

// Evict nlines cache lines at stride apart starting at addr from BOTH L1
// and L2. Uses EvictVA (CSR 0x89F) with dest=10 (L3/DRAM), unlike evict_to_l2
// above (dest=01, L2 only). Needed because both L1 and L2 are incoherent on
// ET-SoC-1 (L2 is per-shire). Caller must FENCE before and WAIT_CACHEOPS
// after. NOTE: nlines is encoded in a 4-bit field (max 16). DO NOT pass
// nlines > 16 - use evict_region_past_l2() below for larger regions.
static inline void __attribute__((always_inline)) evict_past_l2(const void * addr, uint64_t nlines, uint64_t stride) {
    uint64_t csr_val = (0x2ULL << 58) | ((uint64_t) addr & 0xFFFFFFFFFFC0ULL) | ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x89F, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory");
}

// Evict a contiguous region from both L1 and L2 so subsequent loads fetch
// from L3/DRAM. Handles regions larger than the 16-line hardware limit by
// issuing multiple evict_past_l2 calls.
static void evict_region_past_l2(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }

    const uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    for (uint64_t off = 0; off < nlines; off += 16) {
        uint64_t batch = nlines - off;
        if (batch > 16) {
            batch = 16;
        }
        evict_past_l2((const void *) (base + off * CL), batch, CL);
    }
}

//******************************************************************************
// Counter signaling between harts via L2 scratchpad (SCP)
//******************************************************************************

// Signal a counter value to the other hart via L2 SCP.
static inline void __attribute__((always_inline))
scp_signal(volatile uint32_t *flag, uint32_t value) {
    FENCE;
    *flag = value;
    FENCE;
    evict_to_l2((const void *)flag, 1, 64);
    WAIT_CACHEOPS;
    FENCE;
}

// Wait for a counter in L2 SCP to reach the expected value.
static inline void __attribute__((always_inline))
scp_wait(volatile uint32_t *flag, uint32_t expected) {
    while (1) {
        evict_to_l2((const void *)flag, 1, 64);
        WAIT_CACHEOPS;
        FENCE;
        if (*flag >= expected) {
            FENCE;
            return;
        }
    }
}

#endif // PLATFORM_H
