#include "csrc/green_ctx/initializer.hpp"
#include <cstdio>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace green_streams {

    // Debug function.
    void check(CUresult r, const char* expression) {
        if (r == CUDA_SUCCESS) return;
        const char *name = nullptr, *message = nullptr;
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &message);
        throw std::runtime_error(std::string(expression) + ": " +
            (name ? name : "unknown CUDA error") + " (" + std::to_string(r) +
            "): " + (message ? message : "no description"));
    }
    #define GC_CHECK(expr) ::green_streams::check((expr), #expr)



    namespace {


        void report(CUresult r, const char* operation) noexcept {
        if (r != CUDA_SUCCESS && r != CUDA_ERROR_DEINITIALIZED)
            std::fprintf(stderr, "green_streams cleanup: %s failed (%d)\n", operation,
                        static_cast<int>(r));
        }


        // Driver operations temporarily use the primary context and restore the
        // caller's context, including the case where no context was current.
        // This is an RAII-style design, where we use obj construction to control
        // cuCtxPushCurrent and destruction to cuCtxPopCurrent .
        class ContextScope {
            public:
            explicit ContextScope(CUcontext ctx) { GC_CHECK(cuCtxPushCurrent(ctx)); }
            ~ContextScope() noexcept {
                CUcontext popped = nullptr;
                report(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
            }
            ContextScope(const ContextScope&) = delete;
            ContextScope& operator=(const ContextScope&) = delete;
        };


    }// end of anonymous namespace



    // RAII-style wrapper for device id, cuda device, and cuda context.
    /*
    It accomplishes:
    (1) Determine which GPU we are using
    (2) Obtain the CUdevice of this particular GPU
    (3) Retain the primary context of this GPU
    (4) Record the process that constructs the DeviceState to prevent missuse of CUDA handles after fork
    */
    struct DeviceState {

        CUdevice device = 0;
        CUcontext primary = nullptr; // The primary cuda context
        int index;
        pid_t pid = getpid();

        // Constrution function. 
        // Use explict to prevent compiler from implicitly transform an int to DeviceState object.
        explicit DeviceState(int i) : index(i) {
            GC_CHECK(cuInit(0));
            GC_CHECK(cuDeviceGet(&device, i));
            int version = 0;
            GC_CHECK(cuDriverGetVersion(&version));
            if (version < 12060){
                throw std::runtime_error("CUDA driver must support CUDA 12.6 or newer");
            }
            
            GC_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        }


        // Check whether the current process is identical to the process that spawns the DeviceState.
        void check_process() const {
            if (getpid() != pid)
            throw std::runtime_error("CUDA handles cannot be used after fork; "
                                    "create StreamInitializer in the GPU worker");
        }


        ~DeviceState() noexcept {
            if (getpid() == pid && primary)
            report(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
        }
    };


    // Constructor of GreenStream.
    GreenStream::GreenStream(std::shared_ptr<DeviceState> d) : device_(std::move(d)) {}



    // Initialize a GreenStream.
    void GreenStream::initialize(CUdevResource resource, int priority) {
        CUdevResourceDesc desc = nullptr;
        GC_CHECK(cuDevResourceGenerateDesc(&desc, &resource, 1));
        // CUDA 12.6 exposes no cuDevResourceDescDestroy. Do not invent one or free
        // the opaque descriptor with host free(). Provisioned resources are released
        // by cuGreenCtxDestroy.
        GC_CHECK(cuGreenCtxCreate(&green_, desc, device_->device,
                                CU_GREEN_CTX_DEFAULT_STREAM));
        GC_CHECK(cuGreenCtxStreamCreate(&stream_, green_, CU_STREAM_NON_BLOCKING,
                                        priority));
        CUdevResource actual{};
        GC_CHECK(cuGreenCtxGetDevResource(green_, &actual, CU_DEV_RESOURCE_TYPE_SM));
        sm_count_ = actual.sm.smCount;
        if (sm_count_ != resource.sm.smCount){
            throw std::runtime_error("Provisioned SM count differs from split result");
        }
            
        if (!verify_binding()){
            throw std::runtime_error("Stream is not associated with the expected green context");
        }
            
    }




    GreenStream::~GreenStream() noexcept {
        if (getpid() != device_->pid) {
            return;
        }
        const CUresult pushed = cuCtxPushCurrent(device_->primary);
        if (pushed != CUDA_SUCCESS) { 
            report(pushed, "cuCtxPushCurrent"); 
            return; 
        }
        if (stream_) {
            report(cuStreamSynchronize(stream_), "cuStreamSynchronize");
            report(cuStreamDestroy(stream_), "cuStreamDestroy");
        }
        if (green_){
            report(cuGreenCtxDestroy(green_), "cuGreenCtxDestroy");
        }
        CUcontext popped = nullptr;
        report(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
    }



    // Return the cuda stream associated with the GreenStream.
    CUstream GreenStream::stream() const { 
        device_->check_process(); 
        return stream_; 
    }


    // Return the pointer of the cuda stream associated with the GreenStream.
    std::uintptr_t GreenStream::stream_ptr() const {
        return reinterpret_cast<std::uintptr_t>(stream());
    }


    // Return the pointer of the CUDA Green Context.
    std::uintptr_t GreenStream::green_context_ptr() const {
        device_->check_process(); 
        return reinterpret_cast<std::uintptr_t>(green_);
    }


    // Return the device index that the GreenStream resides.
    int GreenStream::device_index() const noexcept { 
        return device_->index; 
    }


    // Synchronize for the GreenStream.
    void GreenStream::synchronize() const {
        device_->check_process(); 
        ContextScope scope(device_->primary);
        GC_CHECK(cuStreamSynchronize(stream_));
    }


    // Query whether or not all jobs preveously submitted to this GreenStream have been completed.
    bool GreenStream::query() const {
        device_->check_process(); 
        ContextScope scope(device_->primary);
        CUresult r = cuStreamQuery(stream_);
        if (r == CUDA_ERROR_NOT_READY) {
            return false;
        }
        GC_CHECK(r); 
        return true;
    }


    // Checks whether or not the GreenStream is associated with the expected green context.
    bool GreenStream::verify_binding() const {
        device_->check_process(); 
        ContextScope scope(device_->primary);
        CUgreenCtx actual = nullptr;
        GC_CHECK(cuStreamGetGreenCtx(stream_, &actual));
        return actual == green_;
    }


    // Initialize the GreenStream Manager.
    // TODO: Currently we only have GreenStreams for Vision Encoding and LLM Decoding. 
    // GreenStreams for LLM Prefilling is pending development.
    StreamInitializer::StreamInitializer(
        int index, 
        const std::vector<unsigned>& sms,
        bool strict, 
        unsigned flags, 
        int dp, 
        int vp
    ){
        if (sms.empty()){
            throw std::invalid_argument("decode_sms must not be empty");
        }
        std::set<unsigned> seen;
        for (unsigned n : sms){
            if (n == 0 || !seen.insert(n).second)
            throw std::invalid_argument("decode_sms must contain unique positive counts");            
        }

        device_ = std::make_shared<DeviceState>(index);
        ContextScope scope(device_->primary);
        CUdevResource all{};
        GC_CHECK(cuDeviceGetDevResource(device_->device, &all, CU_DEV_RESOURCE_TYPE_SM));
        total_sms_ = all.sm.smCount;
        // Validate all scalar requests before creating any contexts.
        for (unsigned n : sms)
            if (n >= total_sms_)
            throw std::invalid_argument("Decode count must leave SMs for Vision");
        partitions_.reserve(sms.size());
        for (unsigned n : sms) {
            CUdevResource decode{}, vision{};
            unsigned groups = 1;
            // CRITICAL: decode and its complement come from ONE split invocation.
            GC_CHECK(cuDevSmResourceSplitByCount(&decode, &groups, &all, &vision, flags, n));
            if (
                groups != 1 || decode.type != CU_DEV_RESOURCE_TYPE_SM ||
                vision.type != CU_DEV_RESOURCE_TYPE_SM || !vision.sm.smCount ||
                decode.sm.smCount + vision.sm.smCount != total_sms_
            ){
                throw std::runtime_error("Driver did not return a complete nonempty partition pair");
            }
            
            if (strict && decode.sm.smCount != n){
                throw std::runtime_error("Requested " + std::to_string(n) +
                " Decode SMs, driver returned " + std::to_string(decode.sm.smCount) +
                ". Use an aligned count or strict=False to accept rounding.");
            }
            
            auto d = std::shared_ptr<GreenStream>(new GreenStream(device_));
            auto v = std::shared_ptr<GreenStream>(new GreenStream(device_));
            d->initialize(decode, dp);
            v->initialize(vision, vp);
            partitions_.push_back(Partition{"p" + std::to_string(partitions_.size()) +
                "_d" + std::to_string(d->sm_count()) + "_v" +
                std::to_string(v->sm_count()), n, std::move(d), std::move(v)});
        }
    }


    // Return the device index the GreenStream Manager resides.
    int StreamInitializer::device_index() const noexcept { return device_->index; }


    // Synchronize the GreenStreams for all the partition plans.
    // TODO: This function also does not contain for the LLM Prefilling, which is pending development.
    void StreamInitializer::synchronize() const {
        device_->check_process();
        for (const auto& p : partitions_) {
            p.decode->synchronize(); 
            p.vision->synchronize();
        }
    }
}  // namespace green_streams
