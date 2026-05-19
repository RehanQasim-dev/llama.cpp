#pragma once

#include <runtime/IRuntime.h>
#include <runtime/IProfiler.h>
#include <device-layer/IDeviceLayer.h>
#include <string>
#include <unordered_map>
#include <map>
#include <fstream>
#include <vector>
#include <cstdint>
#include "ggml-backend-impl.h"
#include "ggml-et-uberkernel-common.h"
#include "ggml-et.h"


std::shared_ptr<rt::IRuntime> ggml_et_runtime();

struct ggml_backend_et_buffer_type_context {
    int devidx;
    std::string name;
};

struct ggml_backend_et_buffer_context {
    int devidx;
    void * data;                    // Device memory pointer
    size_t size;
    rt::DeviceId rtid;
};

struct ggml_backend_et_context {
    int devidx;
};
struct et_shire_stats_accum_t {
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t l2_misses = 0;
    uint64_t l2_reads = 0;
    uint64_t l2_writes = 0;
    bool active = false;
};

struct et_ms_stats_accum_t {
    uint64_t ms0 = 0;
    uint64_t ms1 = 0;
    bool active = false;
};

struct et_kernel_profile_accum_t {
    uint32_t runs = 0;
    std::map<uint16_t, et_shire_stats_accum_t> shires;
    std::map<uint8_t, et_ms_stats_accum_t> ms;
};

struct ggml_backend_et_device_context;

struct ggml_backend_et_uberkernel_context {
    bool failed = false;
    uint64_t shire_mask = 0;

    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte> params_blob;

    std::byte * device_insts = nullptr;
    std::byte * device_params = nullptr;
    size_t device_insts_capacity = 0;
    size_t device_params_capacity = 0;
};

struct ggml_backend_et_device_context {
    int devidx;
    rt::DeviceId rtid;
    std::string name;
    std::string desc;
    size_t total_mem;
    ggml_backend_buffer_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte* trace_buffer;

    // Profiling
    bool profiling_enabled = false;
    struct ggml_et_profile_stats last_stats;
    struct ggml_et_profile_stats last_stats_sum;
    std::map<std::string, et_kernel_profile_accum_t> profile_accumulators;

    bool uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

struct ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
