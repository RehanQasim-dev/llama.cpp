#include "ggml-et-kernels.h"
#include "ggml-impl.h"
#include "ggml-et-kernels-embed.hpp"
#include "ggml-et-uberkernel-kernel-map.h"
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <algorithm>

#define ET_TRACE_DECODER_IMPL
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

static constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

static size_t ggml_et_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    if (current_capacity == 0) {
        return required_capacity;
    }

    size_t next_capacity = current_capacity;
    while (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    return next_capacity;
}

static void ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx) {
        return;
    }

    uk_ctx->shire_mask = 0;
    uk_ctx->insts.clear();
    uk_ctx->params_blob.clear();
}

static bool ggml_et_uberkernel_ensure_device_capacity(ggml_backend_et_uberkernel_context * uk_ctx,
                                                       ggml_backend_et_device_context * dev_ctx,
                                                       size_t insts_size,
                                                       size_t params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!uk_ctx || !dev_ctx || !runtime) {
        return false;
    }

    try {
        if (uk_ctx->device_insts == nullptr || insts_size > uk_ctx->device_insts_capacity) {
            const size_t new_capacity = ggml_et_next_capacity(uk_ctx->device_insts_capacity, insts_size);
            if (uk_ctx->device_insts) {
                runtime->freeDevice(dev_ctx->rtid, uk_ctx->device_insts);
            }
            uk_ctx->device_insts = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            uk_ctx->device_insts_capacity = uk_ctx->device_insts ? new_capacity : 0;
        }

        if (uk_ctx->device_params == nullptr || params_size > uk_ctx->device_params_capacity) {
            const size_t new_capacity = ggml_et_next_capacity(uk_ctx->device_params_capacity, params_size);
            if (uk_ctx->device_params) {
                runtime->freeDevice(dev_ctx->rtid, uk_ctx->device_params);
            }
            uk_ctx->device_params = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            uk_ctx->device_params_capacity = uk_ctx->device_params ? new_capacity : 0;
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        return false;
    }

    return uk_ctx->device_insts != nullptr && uk_ctx->device_params != nullptr;
}

// Get embedded kernel data by name
static std::vector<std::byte> ggml_et_get_embedded_kernel(const std::string& kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    if (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        return {};
    }

    const unsigned char* data = it->second.first;
    uint64_t size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    return buffer;
}

// Read kernel from file (for development/override)
static std::vector<std::byte> ggml_et_read_kernel_file(const std::string& kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
}

// Load kernel from file or embedded data
bool ggml_et_load_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        return false;
    }

    // Check if kernel already loaded
    if (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        return true;
    }

    std::vector<std::byte> kernel_data;
    const char* kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    if (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data = ggml_et_read_kernel_file(kernel_file);

        if (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } else {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    if (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        if (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            return false;
        }
    }

    try {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        return true;

    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        return false;
    }
}

static bool ggml_et_launch_kernel_internal(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name,
                                           void* params, size_t params_size, uint64_t shire_mask, bool enable_print,
                                           bool sync_error_check) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        return false;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        if (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            return false;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        if (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            return false;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    try {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(true);          // Wait for completion
        k_opts.setFlushL3(false);         // No L3 flush needed
        if(enable_print  || dev_ctx->profiling_enabled) {
            k_opts.setUserTracing(
                reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                0,                              // threshold
                shire_mask,                     // shire mask
                0xFFFFFFFFFFFFFFFFULL,          // threadMask — all threads
                0xFFFFFFFFU,                    // eventMask — all events
                0xFFFFFFFFU                     // filterMask — all levels
            );
        }

        if(sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            if(!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                for(const auto& error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (int)error.errorCode_);
                }
                abort();
            }
        }

        runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                             reinterpret_cast<std::byte*>(params), params_size, k_opts);

        if(enable_print  || dev_ctx->profiling_enabled) {
            std::vector<std::byte> hostTraceBuf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(
                dev_ctx->default_stream, dev_ctx->trace_buffer, hostTraceBuf.data(), ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            const auto* traceHeader = reinterpret_cast<const trace_buffer_std_header_t*>(hostTraceBuf.data());
            const trace_entry_header_t* entry = nullptr;

            // Profiling stats
            struct compute_packet_t {
                uint64_t cycle;
                uint64_t hpm3, hpm4, hpm5, hpm6, hpm7, hpm8;
                bool operator<(const compute_packet_t& o) const { return cycle < o.cycle; }
            };
            std::map<uint16_t, std::vector<compute_packet_t>> compute_even;
            std::map<uint16_t, std::vector<compute_packet_t>> compute_odd;

            struct sc_packet_t {
                uint64_t cycle;
                uint64_t sc0, sc1;
                bool operator<(const sc_packet_t& o) const { return cycle < o.cycle; }
            };
            std::map<uint16_t, std::vector<sc_packet_t>> sc_packets;

            struct ms_packet_t {
                uint64_t cycle;
                uint64_t ms0, ms1;
                bool operator<(const ms_packet_t& o) const { return cycle < o.cycle; }
            };
            std::map<uint8_t, std::vector<ms_packet_t>> ms_packets;

            while ((entry = Trace_Decode(traceHeader, entry))) {
                if (entry->type == TRACE_TYPE_STRING && enable_print) {
                    const auto* strEntry = reinterpret_cast<const trace_string_t*>(entry);
                    printf("[hart %d] %s", entry->hart_id, strEntry->string);
                } else if (entry->type == TRACE_TYPE_PMC_COUNTERS_COMPUTE) {
                    const auto* pmc = reinterpret_cast<const trace_pmc_counters_compute_t*>(entry);
                    uint16_t neigh_id = entry->hart_id >> 4;
                    compute_packet_t p = {entry->cycle, pmc->hpmcounter3, pmc->hpmcounter4, pmc->hpmcounter5, pmc->hpmcounter6, pmc->hpmcounter7, pmc->hpmcounter8};
                    if ((entry->hart_id % 2) == 0) {
                        compute_even[neigh_id].push_back(p);
                    } else {
                        compute_odd[neigh_id].push_back(p);
                    }
                } else if (entry->type == TRACE_TYPE_PMC_COUNTERS_SC) {
                    const auto* sc = reinterpret_cast<const trace_pmc_counters_sc_t*>(entry);
                    uint16_t neigh_id = entry->hart_id >> 4; // 16 threads per neighborhood
                    sc_packets[neigh_id].push_back({entry->cycle, sc->sc_pmc0, sc->sc_pmc1});
                } else if (entry->type == TRACE_TYPE_PMC_COUNTERS_MS) {
                    const auto* ms = reinterpret_cast<const trace_pmc_counters_ms_t*>(entry);
                    ms_packets[ms->ms_id].push_back({entry->cycle, ms->ms_pmc0, ms->ms_pmc1});
                }
            }

            if (dev_ctx->profiling_enabled) {
                auto& accum = dev_ctx->profile_accumulators[kernel_name];
                accum.runs++;

                std::map<uint16_t, uint64_t> max_cycles_per_shire;

                for (auto& kv : compute_even) {
                    uint16_t neigh_id = kv.first;
                    uint16_t shire_id = neigh_id >> 2;
                    auto& ss = accum.shires[shire_id];
                    ss.active = true;

                    auto& vec = kv.second;
                    if (vec.size() >= 2) {
                        std::sort(vec.begin(), vec.end());
                        auto& end_p = vec[vec.size() - 1];
                        auto& start_p = vec[vec.size() - 2];

                        uint64_t cycles = end_p.hpm3 - start_p.hpm3;
                        if (cycles > max_cycles_per_shire[shire_id]) {
                            max_cycles_per_shire[shire_id] = cycles;
                        }

                        ss.instructions += (end_p.hpm4 + end_p.hpm5) - (start_p.hpm4 + start_p.hpm5);
                        ss.l2_misses += end_p.hpm6 - start_p.hpm6;
                    }
                }

                for (auto& kv : compute_odd) {
                    uint16_t neigh_id = kv.first;
                    uint16_t shire_id = neigh_id >> 2;
                    auto& ss = accum.shires[shire_id];
                    ss.active = true;

                    auto& vec = kv.second;
                    if (vec.size() >= 2) {
                        std::sort(vec.begin(), vec.end());
                        auto& end_p = vec[vec.size() - 1];
                        auto& start_p = vec[vec.size() - 2];
                        ss.instructions += (end_p.hpm4 + end_p.hpm5) - (start_p.hpm4 + start_p.hpm5);
                        ss.l2_misses += end_p.hpm6 - start_p.hpm6;
                    }
                }

                for (const auto& kv : max_cycles_per_shire) {
                    accum.shires[kv.first].cycles += kv.second;
                }

                for (auto& kv : sc_packets) {
                    uint16_t shire_id = kv.first >> 2;
                    auto& ss = accum.shires[shire_id];
                    ss.active = true;
                    auto& vec = kv.second;
                    if (vec.size() >= 2) {
                        std::sort(vec.begin(), vec.end());
                        auto& end_p = vec[vec.size() - 1];
                        auto& start_p = vec[vec.size() - 2];
                        ss.l2_reads += end_p.sc0 - start_p.sc0;
                        ss.l2_writes += end_p.sc1 - start_p.sc1;
                    }
                }

                for (auto& kv : ms_packets) {
                    auto& mss = accum.ms[kv.first];
                    mss.active = true;
                    auto& vec = kv.second;
                    if (vec.size() >= 2) {
                        std::sort(vec.begin(), vec.end());
                        auto& end_p = vec[vec.size() - 1];
                        auto& start_p = vec[vec.size() - 2];
                        mss.ms0 += end_p.ms0 - start_p.ms0;
                        mss.ms1 += end_p.ms1 - start_p.ms1;
                    }
                }
            }
        }

        if(sync_error_check) {
            // Already triggered. No need to retrigger
            if(!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            if(!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                for(const auto& error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (int)error.errorCode_);
                }
                abort();
            }
        }

        return true;
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        return false;
    }
}

void ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx) {
        return;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

static bool ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context * dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx || !dev_ctx) {
        return false;
    }

    if (uk_ctx->insts.empty()) {
        return true;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = true;
        return false;
    }

    const size_t insts_size = uk_ctx->insts.size() * sizeof(ggml_et_uberkernel_inst);
    const size_t params_size = uk_ctx->params_blob.size();
    bool ok = false;

    try {
        if (!ggml_et_uberkernel_ensure_device_capacity(uk_ctx, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = true;
            ggml_et_uberkernel_reset_segment(uk_ctx);
            return false;
        }

        runtime->memcpyHostToDevice(dev_ctx->default_stream,
                                    reinterpret_cast<const std::byte *>(uk_ctx->insts.data()),
                                    uk_ctx->device_insts, insts_size, true);
        runtime->memcpyHostToDevice(dev_ctx->default_stream,
                                    uk_ctx->params_blob.data(),
                                    uk_ctx->device_params, params_size, true);
        runtime->waitForStream(dev_ctx->default_stream);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(uk_ctx->insts.size()),
            static_cast<uint32_t>(sizeof(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(uk_ctx->device_insts),
            reinterpret_cast<uint64_t>(uk_ctx->device_params),
        };

        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, sizeof(params), uk_ctx->shire_mask, false, false);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;
    ggml_et_uberkernel_reset_segment(uk_ctx);
    return ok;
}

void ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx) {
        return;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

bool ggml_et_uberkernel_failed(const ggml_backend_et_uberkernel_context * uk_ctx) {
    return uk_ctx && uk_ctx->failed;
}

static bool ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      const std::string & kernel_name,
                                      void * params,
                                      size_t params_size,
                                      uint64_t shire_mask,
                                      bool enable_print,
                                      bool sync_error_check) {
    if (!dev_ctx) {
        return false;
    }

    ggml_backend_et_uberkernel_context * uk_ctx = &dev_ctx->uberkernel;
    const uint16_t uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    if (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        if (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            return false;
        }
        return ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print, sync_error_check);
    }

    const size_t params_offset = ggml_et_align_up(uk_ctx->params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    if (params_offset > uk_ctx->params_blob.size()) {
        uk_ctx->params_blob.resize(params_offset);
    }

    const std::byte * params_bytes = reinterpret_cast<const std::byte *>(params);
    uk_ctx->params_blob.insert(uk_ctx->params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    uk_ctx->insts.push_back(inst);

    if (uk_ctx->insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    return true;
}

bool ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    if (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        return true;
    }

    return ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

bool ggml_et_launch_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name,
                          void* params, size_t params_size, uint64_t shire_mask, bool enable_print,
                          bool sync_error_check) {
    if (!dev_ctx) {
        return false;
    }

    if (!dev_ctx->uberkernel_enabled) {
        return ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print, sync_error_check);
    }

    return ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print, sync_error_check);
}

void ggml_et_unload_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it != dev_ctx->loaded_kernels.end()) {
        try {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } catch (const std::exception& e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

void ggml_et_unload_all_kernels(ggml_backend_et_device_context* dev_ctx) {
    if (!dev_ctx) {
        return;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    for (const auto& kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    for (const auto& kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context* dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    for (const auto& kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    return loaded_kernels;
}


void ggml_et_dump_and_reset_profile(ggml_backend_et_device_context* dev_ctx) {
    if (dev_ctx->profile_accumulators.empty()) return;

    std::string filename = "et_events.csv";
    {
        std::ifstream probe(filename);
        if (probe.good()) {
            for (int i = 1; ; ++i) {
                std::string candidate = "et_events_" + std::to_string(i) + ".csv";
                std::ifstream p2(candidate);
                if (!p2.good()) { filename = candidate; break; }
            }
        }
    }

    std::ofstream ofs(filename);
    if (ofs.is_open()) {
        ofs << "Kernel,Type,ID,Cycles,Instructions,IPC,L2Miss,SC_Reads,SC_Writes,MS_Reads,MS_Writes\n";
        for (const auto& kv : dev_ctx->profile_accumulators) {
            const auto& kernel_name = kv.first;
            const auto& accum = kv.second;
            if (accum.runs == 0) continue;

            for (const auto& sh : accum.shires) {
                if (sh.second.active) {
                    uint64_t cycles = sh.second.cycles / accum.runs;
                    uint64_t inst   = sh.second.instructions / accum.runs;
                    double   ipc    = cycles > 0 ? (double)inst / (double)cycles : 0.0;
                    ofs << kernel_name << ",Shire," << sh.first << ","
                        << cycles << ","
                        << inst << ","
                        << ipc << ","
                        << (sh.second.l2_misses / accum.runs) << ","
                        << (sh.second.l2_reads / accum.runs) << ","
                        << (sh.second.l2_writes / accum.runs) << ",,\n";
                }
            }
            for (const auto& ms : accum.ms) {
                if (ms.second.active) {
                    ofs << kernel_name << ",MemShire," << (int)ms.first << ",,,,,,,"
                        << (ms.second.ms0 / accum.runs) << ","
                        << (ms.second.ms1 / accum.runs) << "\n";
                }
            }
        }
    }

    dev_ctx->profile_accumulators.clear();
}
