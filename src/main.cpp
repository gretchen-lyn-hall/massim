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
  py::enum_<massim::PickMassMode>(m, "PickMassMode")
      .value("PICK_BY_MASS", massim::PICK_BY_MASS)
      .value("PICK_BY_FREQ", massim::PICK_BY_FREQ)
      .value("PICK_BY_WEIGHT", massim::PICK_BY_WEIGHT)
      .export_values();

  
  py::class_<massim::TransformSelection>(m, "TransformSelection")
      .def_readonly("delta_m", &massim::TransformSelection::delta_m)
      .def_readonly("xfrm_id", &massim::TransformSelection::xfrm_id)
      .def_readonly("fwd", &massim::TransformSelection::forward_len)
      .def_readonly("back", &massim::TransformSelection::back_len);



  py::class_<massim::TransformList>(m, "TransformList")
      .def(py::init<massim::ConstArrayRef, massim::ConstArrayRef, massim::ConstArrayRef,
                    massim::ConstArrayRef, bool, bool>(),
           "", "masses", "weights", "mean_lens", "mean_lens_stds", "same_len",
           "always_center")
      .def("choose_transform",
           py::overload_cast<const massim::BoolMatType&, massim::RNG&>(
               &massim::TransformList::choose_transform, py::const_),
           "", py::arg("presence"), py::arg("rng"))
      .def("pick_lens", &massim::TransformList::pick_lens, "",
	   py::arg("mean_len"), py::arg("size"), py::arg("rng"));

  py::class_<massim::TransformTracker>(m, "TransformTracker")
      .def(py::init<massim::ConstArrayRef, double>(),
           "", "mass_deltas", "tolerance_ppm")
      .def("contains", &massim::TransformTracker::contains, "",
	   py::arg("test_mass"))
      .def("add_mass", &massim::TransformTracker::add_mass, "",
	   py::arg("mass"))
      .def("counts", &massim::TransformTracker::counts, "")
      .def("total_count", &massim::TransformTracker::total_count, "")
      .def("masses", &massim::TransformTracker::masses, "");
  
  py::class_<massim::TandemTransformer>(m, "TandemTransformer")
      .def(py::init < massim::ConstMatRef, massim::ConstArrayRef,
           const massim::TransformList &, const massim::Distribution &,
           const massim::Distribution &, double, double, double, double,
	   massim::PickMassMode, double, double>())
    .def("apply_transforms", &massim::TandemTransformer::apply_transforms,
	    "doc", py::arg("rng"), py::arg("max_new_peaks")=0, py::arg("target_masses")=0);
  py::class_<massim::TandemTransformer::TransformerResult>(m, "TransformerResult")
    .def("weight_peak", &massim::TandemTransformer::TransformerResult::weight_peak,
	 "doc")
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


  py::class_<massim::ProfileGenerator>(m, "ProfileGenerator")
      .def(py::init<int, const massim::TransformList &,
                    const massim::Distribution &,
                    double, double, double, double, massim::PickMassMode,
	   double, double>(), "doc",
           py::arg("num_profiles"), py::arg("transforms"),
           py::arg("intensity_scale"), 
           py::arg("mass_min") = 150.0, py::arg("mass_max") = 1200.0,
           py::arg("min_intensity") = 1e-6,
           py::arg("thresh_ppm") = 1.0,
           py::arg("mass_mode") = massim::PICK_BY_FREQ,
	   py::arg("mass_center")=600.0,
	   py::arg("mass_scale")=100.0
           )
      .def("apply_transforms", &massim::ProfileGenerator::apply_transforms,
           "doc", py::arg("rng"), py::arg("targ_components") = 0,
           py::arg("target_masses") = 0)
      .def("weight_peak", &massim::ProfileGenerator::weight_peak, "doc",
	   py::arg("mass"))
      .def("set_component", &massim::ProfileGenerator::set_component, "doc",
           py::arg("profile_id"), py::arg("mass"), py::arg("weight"),
	   py::arg("overwrite")=true)
      .def("profiles", &massim::ProfileGenerator::profiles,
	 "doc")
    .def("masses", &massim::ProfileGenerator::masses,
	 "doc")
    .def("num_profiles", &massim::ProfileGenerator::num_profiles,
	"doc")
    .def("num_masses", &massim::ProfileGenerator::num_masses,
	"doc")
    .def("num_components", &massim::ProfileGenerator::num_components,
	 "doc")
    .def("stats", &massim::ProfileGenerator::stats,
	 "doc");

  py::class_<massim::ProfileGenerator::ProfileResult>(m, "ProfileResult")
      .def_readonly("indices", &massim::ProfileGenerator::ProfileResult::indices)
      .def_readonly("weights", &massim::ProfileGenerator::ProfileResult::weights);

  

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
  m.def("find_transforms", &massim::find_transforms, "", py::arg("masses"),
        py::arg("xfrm_masses"), py::arg("err_abs"),
        py::arg("allow_overlap") = true);
  m.def("diffsort", &massim::diffsort, "", py::arg("masses"));
  m.def("simplediffsort", &massim::simplediffsort, "", py::arg("masses"));

  


}
