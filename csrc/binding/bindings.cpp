#include "csrc/green_ctx/initializer.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace green_streams;

#ifdef GREEN_STREAMS_WITH_PROBE
void launch_probe(std::uintptr_t stream, std::uintptr_t output,
                  unsigned blocks, unsigned long long hold_ns);
#endif

PYBIND11_MODULE(_native, m) {
  m.doc() = "CUDA 12.6 Driver API Green Context stream initializer";
  m.attr("cuda_headers_version") = CUDA_VERSION;
  m.attr("IGNORE_SM_COSCHEDULING") =
      static_cast<unsigned>(CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING);
  py::class_<GreenStream, std::shared_ptr<GreenStream>>(m, "GreenStream")
      .def_property_readonly("stream_ptr", &GreenStream::stream_ptr)
      .def_property_readonly("green_context_ptr", &GreenStream::green_context_ptr)
      .def_property_readonly("sm_count", &GreenStream::sm_count)
      .def_property_readonly("device_index", &GreenStream::device_index)
      .def("synchronize", &GreenStream::synchronize,
           py::call_guard<py::gil_scoped_release>())
      .def("query", &GreenStream::query)
      .def("verify_binding", &GreenStream::verify_binding);
  py::class_<Partition>(m, "Partition")
      .def_readonly("id", &Partition::id)
      .def_readonly("requested_decode_sms", &Partition::requested_decode_sms)
      .def_readonly("decode", &Partition::decode)
      .def_readonly("vision", &Partition::vision);
  py::class_<StreamInitializer>(m, "StreamInitializer")
      .def(py::init<int, const std::vector<unsigned>&, bool, unsigned, int, int>(),
           py::arg("device_index"), py::arg("decode_sms"),
           py::arg("strict") = true, py::arg("split_flags") = 0,
           py::arg("decode_priority") = 0, py::arg("vision_priority") = 0)
      .def_property_readonly("partitions", [](const StreamInitializer& x) {
        return x.partitions();  // value copies retain shared ownership of streams
      })
      .def_property_readonly("total_sms", &StreamInitializer::total_sms)
      .def_property_readonly("device_index", &StreamInitializer::device_index)
      .def("synchronize", &StreamInitializer::synchronize,
           py::call_guard<py::gil_scoped_release>());


           
#ifdef GREEN_STREAMS_WITH_PROBE
  // Internal raw-pointer boundary. Public Python wrapper validates the tensor.
  m.def("_launch_probe", [](const std::shared_ptr<GreenStream>& s,
                            std::uintptr_t output, unsigned blocks,
                            unsigned long long hold_ns) {
    launch_probe(s->stream_ptr(), output, blocks, hold_ns);
  });
#endif
}
