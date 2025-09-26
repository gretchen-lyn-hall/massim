#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "cpp/common.h"
#include "cpp/rng.h"
#include "cpp/distribution.h"
#include "cpp/utils.h"
#include "cpp/spectra.h"
#include "cpp/fft_transformer.h"
#include "cpp/gradient_response.h"
#include "cpp/metrics.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for Massim";
  m.def("parse_dist", &massim::parse_dist, "Guess?", py::arg("txt"));
  m.def("test_massim", &massim::test_massim, "Guess?");
  py::class_<massim::GradientResponse>(m, "GradientResponse")
    .def("apply", &massim::GradientResponse::apply);

  
  m.def("gen_dist", &massim::gen_dist,
      "Generate a C-compatible probability dist",
      py::arg("dist_type"), py::arg("args"));

  m.def("adjust_masses", &massim::adjust_masses,
      "Align masses in a numpy array to the nearest valies in a list of valid masses",
      py::arg("masses").noconvert(), py::arg("valid_masses"));
  // RNG / Distribution
  py::class_<massim::RNG>(m, "RNG")
    .def(py::init<>())
    .def(py::init<int>())
    .def(py::init<massim::RNG&>())
    .def("__call__", &massim::RNG::operator());


  py::class_<massim::Distribution>(m, "Distribution")
    .def("__call__", [](massim::Distribution& self){return self();})
    .def("__call__", [](massim::Distribution& self, int count){return self(count);})
    .def("__call__", [](massim::Distribution& self,
	    int count,
	    massim::RNG rng){return self(count, &rng);})
    .def("clone", &massim::Distribution::clone);

  // Transformation
  py::class_<massim::TransformSelection>(m, "TransformSelection")
    .def_readonly("delta_m", &massim::TransformSelection::delta_m)
    .def_readonly("xfrm_id", &massim::TransformSelection::xfrm_id)
    .def_readonly("fwd", &massim::TransformSelection::forward_len)
    .def_readonly("back", &massim::TransformSelection::back_len);

  py::class_<massim::TransformList>(m, "TransformList")
      .def(py::init<massim::ArrayRef, massim::ArrayRef, massim::ArrayRef,
	   massim::ArrayRef, bool, bool>(),
           "", "masses", "weights", "mean_lens", "mean_lens_stds",
	   "same_len", "always_center")
    .def("choose_transform", &massim::TransformList::choose_transform, "",
	py::arg("presence"), py::arg("rng"))
    .def("pick_lens", &massim::TransformList::pick_lens, "",
	py::arg("mean_len"), py::arg("size"), py::arg("rng"));

  py::class_<massim::TandemTransformer>(m, "TandemTransformer")
      .def(py::init<massim::ConstMatRef,
	   massim::ConstArrayRef,
		  const massim::TransformList&,
		  const massim::Distribution&,
		  const massim::Distribution&,
		  double,
		  double,
		  double,
		  double>())
    .def("apply_transforms", &massim::TandemTransformer::apply_transforms,
	    "doc", py::arg("rng"), py::arg("max_new_peaks")=0, py::arg("target_masses")=0);
  py::class_<massim::TandemTransformer::TransformerResult>(m, "TransformerResult")
    .def("intensity", &massim::TandemTransformer::TransformerResult::intensity,
	 "doc")
    .def("sparse_intensity", &massim::TandemTransformer::TransformerResult::sparse_intensity,
	 "doc")
    .def("masses", &massim::TandemTransformer::TransformerResult::masses,
	 "doc")
    .def("original_ids", &massim::TandemTransformer::TransformerResult::original_ids,
	"doc")
    .def("extend_matrix", &massim::TandemTransformer::TransformerResult::extend_matrix,
	 "Create output matrix using the columns from the input",
	py::arg("src_matrix"),
	 py::arg("use_contrib")=false);

  

  // Metrics
  m.def("compute_aff", &massim::compute_aff,
      "Compute the affinity matrix from a similarity matrix",
      py::arg("sim"));
  m.def("jaccard", py::overload_cast<const Eigen::Ref<const massim::BoolMatType>&>(&massim::jaccard),
      "Compute the Jaccard similarity of a presence/absence matrix",
      py::arg("presence"));
  m.def("braycurtis", &massim::braycurtis,
      "Compute the BrayCurtis similarity of an intensity matrix",
      py::arg("mat"));

  // Utils
  m.def("find_transforms", &massim::find_transforms, "",
      py::arg("masses"), py::arg("xfrm_masses"), py::arg("err_abs"));


}
