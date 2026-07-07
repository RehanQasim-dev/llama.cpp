#include "ggml-et-profile.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

// Trace decoder declarations. The single ET_TRACE_DECODER_IMPL definition lives
// in ggml-et-kernels.cpp; here we only use the declarations and link against it.
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

// -----------------------------------------------------------------------------
// Per-launch decoded stats (ported from the et-kernel-bench launcher).
// -----------------------------------------------------------------------------

// Per-compute-shire counters collected from one kernel run.
struct shire_stats {
    uint64_t cycles = 0, instructions = 0, l2_misses = 0;
    uint64_t l2_reads = 0, l2_writes = 0;
    bool active = false;
};

// DRAM (Memory Shire) bandwidth in GB/s.
struct ms_stats {
    double read_bw_gbps  = 0.0;
    double write_bw_gbps = 0.0;
    double total_bw_gbps() const { return read_bw_gbps + write_bw_gbps; }
};

struct run_result {
    std::map<uint16_t, shire_stats> shires;
    std::map<uint8_t,  ms_stats>    ms;
};

// Running total across all launches of one kernel, plus the run count.
struct et_profile_accum {
    std::map<uint16_t, shire_stats> sum_shires;
    std::map<uint8_t,  ms_stats>    sum_ms;
    int runs = 0;
};

static bool                                g_active = false;
static std::map<std::string, et_profile_accum> g_accum;

// Memory-Shire PMCs are 40-bit and may wrap during a run.
static inline uint64_t calc_delta_40bit(uint64_t beg, uint64_t end) {
    return (end >= beg) ? (end - beg) : (((1ULL << 40) - beg) + end);
}

// -----------------------------------------------------------------------------
// Decode one trace buffer into per-shire / per-memory-shire stats.
//
// profile.c emits two snapshots per hart (before + after the kernel).
// Shire mapping: neighbourhood_id = hart_id >> 4; shire_id = nid >> 2.
// -----------------------------------------------------------------------------
static run_result decode_trace(const void * trace_buf) {
    const auto * hdr = reinterpret_cast<const trace_buffer_std_header_t *>(trace_buf);

    struct compute_pkt {
        uint64_t cycle, hpm3, hpm4, hpm5, hpm6, hpm7, hpm8;
        bool operator<(const compute_pkt & o) const { return cycle < o.cycle; }
    };
    struct sc_pkt {
        uint64_t cycle, sc0, sc1;
        bool operator<(const sc_pkt & o) const { return cycle < o.cycle; }
    };
    struct ms_pkt {
        uint64_t cycle, ms0, ms1;
        bool operator<(const ms_pkt & o) const { return cycle < o.cycle; }
    };

    std::map<uint16_t, std::vector<compute_pkt>> even_pkts, odd_pkts;
    std::map<uint16_t, std::vector<sc_pkt>>      sc_pkts;
    std::map<uint8_t,  std::vector<ms_pkt>>      ms_pkts;

    const trace_entry_header_t * entry = nullptr;
    while ((entry = Trace_Decode(hdr, entry))) {
        switch (entry->type) {
            case TRACE_TYPE_PMC_COUNTERS_COMPUTE: {
                const auto * p = reinterpret_cast<const trace_pmc_counters_compute_t *>(entry);
                uint16_t nid = entry->hart_id >> 4;
                compute_pkt pkt{entry->cycle,
                    p->hpmcounter3, p->hpmcounter4, p->hpmcounter5,
                    p->hpmcounter6, p->hpmcounter7, p->hpmcounter8};
                if (entry->hart_id % 2 == 0)
                    even_pkts[nid].push_back(pkt);
                else
                    odd_pkts[nid].push_back(pkt);
                break;
            }
            case TRACE_TYPE_PMC_COUNTERS_SC: {
                const auto * p = reinterpret_cast<const trace_pmc_counters_sc_t *>(entry);
                uint16_t nid = entry->hart_id >> 4;
                sc_pkts[nid].push_back({entry->cycle, p->sc_pmc0, p->sc_pmc1});
                break;
            }
            case TRACE_TYPE_PMC_COUNTERS_MS: {
                const auto * p = reinterpret_cast<const trace_pmc_counters_ms_t *>(entry);
                ms_pkts[p->ms_id].push_back({entry->cycle, p->ms_pmc0, p->ms_pmc1});
                break;
            }
            default:
                break;
        }
    }

    run_result rr;
    std::map<uint16_t, uint64_t> max_cycles;

    for (auto & [nid, vec] : even_pkts) {
        uint16_t sid = nid >> 2;
        rr.shires[sid].active = true;
        if (vec.size() >= 2) {
            std::sort(vec.begin(), vec.end());
            auto & end = vec.back(), & beg = vec[vec.size() - 2];
            uint64_t cyc = end.hpm3 - beg.hpm3;
            if (cyc > max_cycles[sid]) max_cycles[sid] = cyc;
            rr.shires[sid].instructions += (end.hpm4 + end.hpm5) - (beg.hpm4 + beg.hpm5);
            rr.shires[sid].l2_misses    += end.hpm6 - beg.hpm6;
        }
    }
    for (auto & [nid, vec] : odd_pkts) {
        uint16_t sid = nid >> 2;
        rr.shires[sid].active = true;
        if (vec.size() >= 2) {
            std::sort(vec.begin(), vec.end());
            auto & end = vec.back(), & beg = vec[vec.size() - 2];
            rr.shires[sid].instructions += (end.hpm4 + end.hpm5) - (beg.hpm4 + beg.hpm5);
            rr.shires[sid].l2_misses    += end.hpm6 - beg.hpm6;
        }
    }
    for (auto & [sid, cyc] : max_cycles) rr.shires[sid].cycles += cyc;

    for (auto & [nid, vec] : sc_pkts) {
        uint16_t sid = nid >> 2;
        rr.shires[sid].active = true;
        if (vec.size() >= 2) {
            std::sort(vec.begin(), vec.end());
            auto & end = vec.back(), & beg = vec[vec.size() - 2];
            rr.shires[sid].l2_reads  += end.sc0 - beg.sc0;
            rr.shires[sid].l2_writes += end.sc1 - beg.sc1;
        }
    }
    for (auto & [ms_id, vec] : ms_pkts) {
        if (vec.size() < 2) continue;
        std::sort(vec.begin(), vec.end());
        auto & end = vec.back(), & beg = vec[vec.size() - 2];
        uint64_t reads  = calc_delta_40bit(beg.ms0, end.ms0);
        uint64_t writes = calc_delta_40bit(beg.ms1, end.ms1);
        uint64_t cyc    = end.cycle - beg.cycle;   // MemShire clock ~1 GHz
        // GB/s = (requests * 64 B/line) / cycles (bytes/cycle at 1 GHz = GB/s)
        double rbw = (cyc > 0) ? static_cast<double>(reads)  * 64.0 / cyc : 0.0;
        double wbw = (cyc > 0) ? static_cast<double>(writes) * 64.0 / cyc : 0.0;
        rr.ms[ms_id] = { rbw, wbw };
    }
    return rr;
}

// -----------------------------------------------------------------------------
// Transposed console table: one column per hardware unit, one row per stat.
// -----------------------------------------------------------------------------
static void print_table(const char * label, const std::string & kernel_name,
                        const std::map<uint16_t, shire_stats> & avg_shires,
                        const std::map<uint8_t,  ms_stats>    & avg_ms,
                        int runs) {
    std::vector<uint16_t> sids;
    for (auto & [sid, s] : avg_shires) if (s.active) sids.push_back(sid);
    std::vector<uint8_t> mids;
    for (auto & [id, m] : avg_ms) mids.push_back(id);

    printf("  ET perf counters | %s | kernel=%s | averaged over %d run(s)\n",
           label ? label : "", kernel_name.c_str(), runs);

    if (sids.empty()) {
        printf("    (no active compute shires in trace)\n");
    } else {
        printf("    %-13s", "Shire");        for (auto id : sids) printf("%11u", (unsigned) id);                        printf("\n");
        printf("    %-13s", "Cycles");       for (auto id : sids) printf("%11llu", (unsigned long long) avg_shires.at(id).cycles);       printf("\n");
        printf("    %-13s", "Instructions"); for (auto id : sids) printf("%11llu", (unsigned long long) avg_shires.at(id).instructions); printf("\n");
        printf("    %-13s", "IPC");          for (auto id : sids) { const auto & s = avg_shires.at(id);
                                                 printf("%11.2f", (s.cycles > 0) ? (double) s.instructions / s.cycles : 0.0); } printf("\n");
        printf("    %-13s", "L1_DMiss");     for (auto id : sids) printf("%11llu", (unsigned long long) avg_shires.at(id).l2_misses); printf("\n");
        printf("    %-13s", "SC_Reads");     for (auto id : sids) printf("%11llu", (unsigned long long) avg_shires.at(id).l2_reads);  printf("\n");
        printf("    %-13s", "SC_Writes");    for (auto id : sids) printf("%11llu", (unsigned long long) avg_shires.at(id).l2_writes); printf("\n");
    }

    if (!mids.empty()) {
        printf("    %-13s", "MemShire");   for (auto id : mids) printf("%11d", (int) id);                        printf("\n");
        printf("    %-13s", "Read_GBps");  for (auto id : mids) printf("%11.2f", avg_ms.at(id).read_bw_gbps);    printf("\n");
        printf("    %-13s", "Write_GBps"); for (auto id : mids) printf("%11.2f", avg_ms.at(id).write_bw_gbps);   printf("\n");
        printf("    %-13s", "Total_GBps"); for (auto id : mids) printf("%11.2f", avg_ms.at(id).total_bw_gbps()); printf("\n");

        double off_total = 0.0;
        for (auto id : mids) off_total += avg_ms.at(id).total_bw_gbps();
        printf("    Off-chip bandwidth (sum of %zu MemShires): %.2f GB/s\n", mids.size(), off_total);
    }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
bool ggml_et_profile_active() {
    return g_active;
}

void ggml_et_profile_begin() {
    g_accum.clear();
    g_active = true;
}

void ggml_et_profile_accumulate(const std::string & kernel_name,
                                const void * trace_buf, size_t trace_size) {
    if (!g_active || trace_buf == nullptr || trace_size == 0) {
        return;
    }

    run_result rr = decode_trace(trace_buf);
    et_profile_accum & a = g_accum[kernel_name];

    for (auto & [sid, s] : rr.shires) {
        if (!s.active) continue;
        auto & acc = a.sum_shires[sid];
        acc.active        = true;
        acc.cycles       += s.cycles;
        acc.instructions += s.instructions;
        acc.l2_misses    += s.l2_misses;
        acc.l2_reads     += s.l2_reads;
        acc.l2_writes    += s.l2_writes;
    }
    for (auto & [id, st] : rr.ms) {
        a.sum_ms[id].read_bw_gbps  += st.read_bw_gbps;
        a.sum_ms[id].write_bw_gbps += st.write_bw_gbps;
    }
    ++a.runs;
}

void ggml_et_profile_end(const char * label) {
    if (!g_active) {
        return;
    }
    g_active = false;

    for (auto & [kernel_name, a] : g_accum) {
        if (a.runs == 0) continue;

        std::map<uint16_t, shire_stats> avg_shires;
        for (auto & [sid, s] : a.sum_shires) {
            shire_stats avg = s;
            avg.cycles       /= a.runs;
            avg.instructions /= a.runs;
            avg.l2_misses    /= a.runs;
            avg.l2_reads     /= a.runs;
            avg.l2_writes    /= a.runs;
            avg_shires[sid] = avg;
        }
        std::map<uint8_t, ms_stats> avg_ms;
        for (auto & [id, st] : a.sum_ms) {
            avg_ms[id] = { st.read_bw_gbps / a.runs, st.write_bw_gbps / a.runs };
        }

        print_table(label, kernel_name, avg_shires, avg_ms, a.runs);
    }

    g_accum.clear();
}
