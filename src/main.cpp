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
#include "cpp/mass_tracker.h"
#include "cpp/profile_generator.h"

// To allow for the '"param"_a' shorthand
using pybind11::literals::operator""_a;
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
                    double, double, double, double, double,double>(), "doc",
           py::arg("num_profiles"), py::arg("transforms"),
           py::arg("intensity_scale"), 
           py::arg("mass_min") = 150.0, py::arg("mass_max") = 1200.0,
           py::arg("min_intensity") = 1e-6,
           py::arg("thresh_ppm") = 1.0,
	   py::arg("preweight")=1.0,
	   py::arg("weight_exponent")=2.0
           )
    // Getters/Setters
    .def_property("preweight",
		  py::overload_cast<>(&massim::ProfileGenerator::preweight, py::const_),
		  py::overload_cast<double>(&massim::ProfileGenerator::preweight)
      )
    .def_property("weight_exponent",
		  py::overload_cast<>(&massim::ProfileGenerator::weight_exponent, py::const_),
		  py::overload_cast<double>(&massim::ProfileGenerator::weight_exponent)
      )
    .def_property("mass_min",
		  py::overload_cast<>(&massim::ProfileGenerator::mass_min, py::const_),
		  py::overload_cast<double>(&massim::ProfileGenerator::mass_min)
      )
    .def_property("mass_max",
		  py::overload_cast<>(&massim::ProfileGenerator::mass_max, py::const_),
		  py::overload_cast<double>(&massim::ProfileGenerator::mass_max)
      )
    .def_property("term_probs",
		  py::overload_cast<>(&massim::ProfileGenerator::term_probs, py::const_),
		  py::overload_cast<const massim::ArrayType&>(&massim::ProfileGenerator::term_probs)
      )
    .def_property("enable_termination",
		  py::overload_cast<>(&massim::ProfileGenerator::enable_termination, py::const_),
		  py::overload_cast<bool>(&massim::ProfileGenerator::enable_termination)
      )
    // Getters for computed values:
          .def("profiles", &massim::ProfileGenerator::profiles,
	 "doc")
    .def("masses", &massim::ProfileGenerator::masses,
	 "doc")
    .def("num_profiles", &massim::ProfileGenerator::num_profiles,
	"doc")
    .def("num_masses", &massim::ProfileGenerator::num_masses,
	"doc")
    .def("weights", &massim::ProfileGenerator::weights,
	"doc")
    .def("counts", &massim::ProfileGenerator::counts,
	"doc")
    .def("stats", &massim::ProfileGenerator::stats,
	 "doc")
    // Computations:
    .def("apply_transforms", &massim::ProfileGenerator::apply_transforms,
	 "doc", py::arg("targ_components"), py::arg("rng"))
    .def("update_component", &massim::ProfileGenerator::update_component, "doc",
           py::arg("profile_id"), py::arg("mass"), py::arg("intensity"))

    ;

  py::class_<massim::ProfileResult>(m, "ProfileResult")
      .def_readonly("indices", &massim::ProfileResult::indices)
      .def_readonly("intensities", &massim::ProfileResult::intensities);

  // Mass Tracker
  py::enum_<massim::TransformMassMode>(m, "TransformMassMode")
      .value("MODE_STRICT", massim::MODE_STRICT)
      .value("MODE_MODERATE", massim::MODE_MODERATE)
      .value("MODE_LAX", massim::MODE_LAX)
      .export_values();

  py::class_<massim::MassTracker>(m, "MassTracker")
      .def(py::init<massim::ConstArrayRef, double, massim::TransformMassMode,
                    bool, bool>(),
           "mass_deltas"_a, "tolerance_ppm"_a, "mode"_a,
           "strict_count"_a = false, "track_applications"_a = false)
      .def("size", &massim::MassTracker::size, "" )
      .def("find_mass", &massim::MassTracker::find_mass, "", "test_mass"_a)
      .def("find_mass_by_id", &massim::MassTracker::find_mass_by_id, "", "mass_id"_a)
      .def("contains_mass_id", &massim::MassTracker::contains_mass_id, "", "mass_id"_a)
      .def("contains_xfrm_by_id", &massim::MassTracker::contains_xfrm_by_id,
           "", "mass_id"_a, "xfrm_id"_a)
      .def("contains_xfrm_by_mass", &massim::MassTracker::contains_xfrm_by_mass,
           "", "mass"_a, "xfrm_id"_a)
      .def("insert_mass", &massim::MassTracker::insert_mass, "", "mass"_a,
           "force_insert"_a = false, "mass_id"_a = -1)
      .def("insert_masses",
           &massim::MassTracker::template insert_masses<massim::ConstArrayRef>,
           "", "masses"_a,
           "force_insert"_a = false)
      .def("counts", &massim::MassTracker::counts, "")
      .def("applications", &massim::MassTracker::applications, "")
      .def("sorted_applications", &massim::MassTracker::sorted_applications, "")
      .def("total_count", &massim::MassTracker::total_count, "")
      .def("masses", &massim::MassTracker::masses, "")
      .def("mass_ids", &massim::MassTracker::mass_ids, "")
      .def("get_seen", &massim::MassTracker::get_seen, "")
      .def(py::pickle([](const massim::MassTracker &mt) {
        return py::make_tuple(mt.mass_deltas(), mt.tolerance_ppm(),
                              mt.mass_mode(), mt.strict_count(),
                              mt.track_applications(),                              
                              mt.get_massmap(), mt.get_seen(), mt.counts(),
			      mt.applications());
          },
          [](py::tuple t) {
            if (t.size() != 9)
              throw std::runtime_error("Invalid state");

            massim::ArrayType mass_deltas(t[0].cast<massim::ConstArrayRef>());
            double tol_ppm(t[1].cast<double>());
            massim::TransformMassMode mode(t[2].cast<massim::TransformMassMode>());
            bool strict(t[3].cast<bool>());
            bool track_apps(t[4].cast<bool>());

            massim::MassTracker result(mass_deltas, tol_ppm, mode, strict,
                                       track_apps);            
            result.set_state(t[5].cast<std::map<double, size_t>>(),
                             t[6].cast<std::vector<std::pair<size_t, int>>>(),
                             t[7].cast<massim::IntArrayType>(),
                             t[8].cast<massim::MassTracker::applications_t>());
	    return result;
            
	  }
	       )
	  );
  m.def("track_transforms", &massim::track_transforms,
        "Find all transformations occurring in a list of masses.",
	"masses"_a, "mass_deltas"_a, "err_ppm"_a, "err_mode"_a);
  


  
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
