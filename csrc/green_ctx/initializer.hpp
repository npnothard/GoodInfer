#pragma once
#include <cuda.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if CUDA_VERSION < 12060
#error "green_streams requires CUDA Toolkit headers >= 12.6"
#endif

namespace green_streams {

    void check(CUresult result, const char* expression);

    struct DeviceState;

    // One immutable stream owns one green context. Borrowed raw handles remain
    // valid only while a shared_ptr<GreenStream> is retained by the caller.
    class GreenStream final {
    
    public:
        ~GreenStream() noexcept;
        GreenStream(const GreenStream&) = delete;
        GreenStream& operator=(const GreenStream&) = delete;
        CUstream stream() const;
        std::uintptr_t stream_ptr() const;
        std::uintptr_t green_context_ptr() const;
        unsigned sm_count() const noexcept { return sm_count_; }
        int device_index() const noexcept;
        void synchronize() const;
        bool query() const;
        bool verify_binding() const;
        
    private:
        friend class StreamInitializer;
        explicit GreenStream(std::shared_ptr<DeviceState> device);
        void initialize(CUdevResource resource, int priority);
        std::shared_ptr<DeviceState> device_;
        CUgreenCtx green_ = nullptr;
        CUstream stream_ = nullptr;
        unsigned sm_count_ = 0;
    };

    // The data structure for SM partition. 
    // TODO: Currently there is no GreenStream for Prefill, which is pending development.
    struct Partition {
        std::string id;
        unsigned requested_decode_sms;
        std::shared_ptr<GreenStream> decode;
        std::shared_ptr<GreenStream> vision;
    };


    // Create once in the GPU worker after process spawning and device selection.
    // Plans are alternatives, NOT globally disjoint reservations. The application
    // must drain the old plan before launching work under a different plan.
    class StreamInitializer final {
    public:
        StreamInitializer(int device_index, const std::vector<unsigned>& decode_sms,
                        bool strict = true, unsigned split_flags = 0,
                        int decode_priority = 0, int vision_priority = 0);
        StreamInitializer(const StreamInitializer&) = delete;
        StreamInitializer& operator=(const StreamInitializer&) = delete;
        unsigned total_sms() const noexcept { return total_sms_; }
        int device_index() const noexcept;
        const std::vector<Partition>& partitions() const noexcept { return partitions_; }
        void synchronize() const;
    private:
        std::shared_ptr<DeviceState> device_;
        unsigned total_sms_ = 0;
        std::vector<Partition> partitions_;
    };
}  // namespace green_streams