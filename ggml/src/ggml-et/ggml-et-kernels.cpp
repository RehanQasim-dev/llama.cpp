#include "ggml-et-kernels.h"
#include "ggml-impl.h"
#include "ggml-et-kernels-embed.hpp"
#include "ggml-et-uberkernel-kernel-map.h"
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <dirent.h>

#define ET_TRACE_DECODER_IMPL
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

static constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

// ---- Debug kernel-trace logging --------------------------------------------
// All decoded et_printf output is written to et_debug_logs/kernel_print_logs_N.txt
// (created relative to the cwd / repo root). One file per process run: the index
// auto-increments past any existing kernel_print_logs_* files. The file is opened
// lazily on the first launch that actually produces trace data.
static FILE* g_et_trace_log_file   = nullptr;
static bool  g_et_trace_log_opened = false;

static FILE* ggml_et_trace_log_get() {
    if (g_et_trace_log_opened) {
        return g_et_trace_log_file;
    }
    g_et_trace_log_opened = true;

    const char* dir = "et_debug_logs";
    mkdir(dir, 0755); // ignore failure (e.g. already exists)

    int next = 1;
    if (DIR* d = opendir(dir)) {
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            int n = 0;
            if (sscanf(ent->d_name, "kernel_print_logs_%d", &n) == 1 && n >= next) {
                next = n + 1;
            }
        }
        closedir(d);
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/kernel_print_logs_%d.txt", dir, next);
    g_et_trace_log_file = fopen(path, "w");
    if (g_et_trace_log_file) {
        GGML_LOG_INFO("ET: writing kernel trace logs to %s\n", path);
    } else {
        GGML_LOG_ERROR("ET: failed to open trace log file %s\n", path);
    }
    return g_et_trace_log_file;
}

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
        // When tracing is on, flush L3 so the kernel's trace-buffer writes are
        // pushed to DRAM before we copy the buffer back (otherwise some
        // readbacks see stale/partial data).
        k_opts.setFlushL3(enable_print);
        if(enable_print) {
            k_opts.setUserTracing(
                reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                0,                              // threshold
                shire_mask,                     // shire mask
                0x1ULL,                         // threadMask — hart 0 only (gets full buffer slice)
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

        if(enable_print) {
            std::vector<std::byte> hostTraceBuf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(
                dev_ctx->default_stream, dev_ctx->trace_buffer, hostTraceBuf.data(), ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            const auto* traceHeader = reinterpret_cast<const trace_buffer_std_header_t*>(hostTraceBuf.data());
            const trace_entry_header_t* entry = nullptr;
            std::string body;
            char        line[1024];
            while ((entry = Trace_Decode(traceHeader, entry))) {
                if (entry->type != TRACE_TYPE_STRING) {
                    continue;
                }
                const auto* strEntry = reinterpret_cast<const trace_string_t*>(entry);
                snprintf(line, sizeof(line), "[hart %d] %s", entry->hart_id, strEntry->string);
                body += line;
            }
            // Generic: title each block with the kernel name, then its captured
            // output. Only emitted when this launch actually produced trace data,
            // so silent launches add nothing to the file. Works for any kernel
            // that calls et_printf (typically guarded on thread 0).
            if (!body.empty()) {
                if (FILE* lf = ggml_et_trace_log_get()) {
                    fprintf(lf, "\n=== kernel %s ===\n", kernel_name.c_str());
                    fputs(body.c_str(), lf);
                    fflush(lf);
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
