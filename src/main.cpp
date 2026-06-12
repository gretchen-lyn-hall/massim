#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "cpp/common.h"
#include "cpp/rng.h"
#include "cpp/distribution.h"
#include "cpp/utils.h"
#include "cpp/spectra.h"
#include "cpp/mass_transformer.h"
#include "cpp/gradient_response.h"
#include "cpp/metrics.h"
#include "cpp/mass_tracker.h"
#include "cpp/profile_generator.h"

// To allow for the '"param"_a' shorthand
using pybind11::literals::operator""_a;

namespace py = pybind11;
using namespace massim;

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for Massim";
  m.def("parse_dist", &parse_dist, "Guess?", py::arg("txt"));
  m.def("test_massim", &test_massim, "Guess?");
  py::class_<GradientResponse>(m, "GradientResponse")
    .def("apply", &GradientResponse::apply);

  
  m.def("gen_dist", &gen_dist,
      "Generate a C-compatible probability dist",
      py::arg("dist_type"), py::arg("args"));

  m.def("adjust_masses", &adjust_masses,
      "Align masses in a numpy array to the nearest valies in a list of valid masses",
      py::arg("masses").noconvert(), py::arg("valid_masses"));
  // RNG / Distribution
  py::class_<RNG>(m, "RNG")
    .def(py::init<>())
    .def(py::init<int>())
    .def(py::init<RNG&>())
    .def("__call__", &RNG::operator());


  py::class_<Distribution>(m, "Distribution")
    .def("__call__", [](Distribution& self){return self();})
    .def("__call__", [](Distribution& self, int count){return self(count);})
    .def("__call__", [](Distribution& self,
	    int count,
	    RNG rng){return self(count, &rng);})
    .def("clone", &Distribution::clone);

  // Transformation
  py::enum_<PickMassMode>(m, "PickMassMode")
    .value("PICK_BY_MASS", PickMassMode::PICK_BY_MASS)
      .value("PICK_BY_FREQ",  PickMassMode::PICK_BY_FREQ)
      .value("PICK_BY_WEIGHT",  PickMassMode::PICK_BY_WEIGHT)
      .export_values();

  
  py::class_<TransformSelection>(m, "TransformSelection")
      .def_readonly("delta_m", &TransformSelection::delta_m)
      .def_readonly("xfrm_id", &TransformSelection::xfrm_id)
      .def_readonly("fwd", &TransformSelection::forward_len)
      .def_readonly("back", &TransformSelection::back_len);



  py::class_<TransformList>(m, "TransformList")
      .def(py::init<ConstArrayRef, ConstArrayRef, ConstArrayRef,
                    ConstArrayRef, bool, bool>(),
           "", "masses", "weights", "mean_lens", "mean_lens_stds", "same_len",
           "always_center")
      .def("choose_transform",
           py::overload_cast<const BoolMatType&, RNG&>(
               &TransformList::choose_transform, py::const_),
           "", py::arg("presence"), py::arg("rng"))
      .def("pick_lens", &TransformList::pick_lens, "",
	   py::arg("mean_len"), py::arg("size"), py::arg("rng"));

  
  py::class_<TandemTransformer>(m, "TandemTransformer")
      .def(py::init < ConstMatRef, ConstArrayRef,
           const TransformList &, const Distribution &,
           const Distribution &, double, double, double, double,
	   PickMassMode, double, double>())
    .def("apply_transforms", &TandemTransformer::apply_transforms,
	    "doc", py::arg("rng"), py::arg("max_new_peaks")=0, py::arg("target_masses")=0);
  py::class_<TandemTransformer::TransformerResult>(m, "TransformerResult")
    .def("weight_peak", &TandemTransformer::TransformerResult::weight_peak,
	 "doc")
    .def("intensity", &TandemTransformer::TransformerResult::intensity,
	 "doc")
    .def("sparse_intensity", &TandemTransformer::TransformerResult::sparse_intensity,
	 "doc")
    .def("masses", &TandemTransformer::TransformerResult::masses,
	 "doc")
    .def("original_ids", &TandemTransformer::TransformerResult::original_ids,
	"doc")
    .def("extend_matrix", &TandemTransformer::TransformerResult::extend_matrix,
	 "Create output matrix using the columns from the input",
	py::arg("src_matrix"),
	 py::arg("use_contrib")=false);


  py::class_<ProfileGenerator>(m, "ProfileGenerator")
      .def(py::init<int, const TransformList &,
                    const Distribution &,
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
		  py::overload_cast<>(&ProfileGenerator::preweight, py::const_),
		  py::overload_cast<double>(&ProfileGenerator::preweight)
      )
    .def_property("weight_exponent",
		  py::overload_cast<>(&ProfileGenerator::weight_exponent, py::const_),
		  py::overload_cast<double>(&ProfileGenerator::weight_exponent)
      )
    .def_property("mass_min",
		  py::overload_cast<>(&ProfileGenerator::mass_min, py::const_),
		  py::overload_cast<double>(&ProfileGenerator::mass_min)
      )
    .def_property("mass_max",
		  py::overload_cast<>(&ProfileGenerator::mass_max, py::const_),
		  py::overload_cast<double>(&ProfileGenerator::mass_max)
      )
    .def_property("term_probs",
		  py::overload_cast<>(&ProfileGenerator::term_probs, py::const_),
		  py::overload_cast<const ArrayType&>(&ProfileGenerator::term_probs)
      )
    .def_property("enable_termination",
		  py::overload_cast<>(&ProfileGenerator::enable_termination, py::const_),
		  py::overload_cast<bool>(&ProfileGenerator::enable_termination)
      )
    // Getters for computed values:
          .def("profiles", &ProfileGenerator::profiles,
	 "doc")
    .def("masses", &ProfileGenerator::masses,
	 "doc")
    .def("num_profiles", &ProfileGenerator::num_profiles,
	"doc")
    .def("num_masses", &ProfileGenerator::num_masses,
	"doc")
    .def("weights", &ProfileGenerator::weights,
	"doc")
    .def("counts", &ProfileGenerator::counts,
	"doc")
    .def("stats", &ProfileGenerator::stats,
	 "doc")
    // Computations:
    .def("apply_transforms", &ProfileGenerator::apply_transforms,
	 "doc", py::arg("targ_components"), py::arg("rng"))
    .def("update_component", &ProfileGenerator::update_component, "doc",
           py::arg("profile_id"), py::arg("mass"), py::arg("intensity"))

    ;

  py::class_<ProfileResult>(m, "ProfileResult")
      .def_readonly("indices", &ProfileResult::indices)
      .def_readonly("intensities", &ProfileResult::intensities);

  // Mass Tracker
  py::enum_<TransformMassMode>(m, "TransformMassMode")
    .value("MODE_STRICT", TransformMassMode::MODE_STRICT)
      .value("MODE_MODERATE", TransformMassMode::MODE_MODERATE)
      .value("MODE_LAX", TransformMassMode::MODE_LAX)
      .export_values();

  py::class_<MassTracker>(m, "MassTracker")
      .def(py::init<ConstArrayRef, double, TransformMassMode,
                    bool, bool>(),
           "mass_deltas"_a, "tolerance_ppm"_a, "mode"_a,
           "strict_count"_a = false, "track_applications"_a = false)
      .def("size", &MassTracker::size, "" )
      .def("find_mass", &MassTracker::find_mass, "", "test_mass"_a)
      .def("find_mass_by_id", &MassTracker::find_mass_by_id, "", "mass_id"_a)
      .def("contains_mass_id", &MassTracker::contains_mass_id, "", "mass_id"_a)
      .def("contains_xfrm_by_id", &MassTracker::contains_xfrm_by_id,
           "", "mass_id"_a, "xfrm_id"_a)
      .def("contains_xfrm_by_mass", &MassTracker::contains_xfrm_by_mass,
           "", "mass"_a, "xfrm_id"_a)
      .def("insert_mass", &MassTracker::insert_mass, "", "mass"_a,
           "force_insert"_a = false, "mass_id"_a = -1)
      .def("insert_masses",
           &MassTracker::template insert_masses<ConstArrayRef>,
           "", "masses"_a,
           "force_insert"_a = false)
      .def("counts", &MassTracker::counts, "")
      .def("applications", &MassTracker::applications, "")
      .def("sorted_applications", &MassTracker::sorted_applications, "")
      .def("total_count", &MassTracker::total_count, "")
      .def("masses", &MassTracker::masses, "")
      .def("mass_ids", &MassTracker::mass_ids, "")
      .def("get_seen", &MassTracker::get_seen, "")
      .def(py::pickle([](const MassTracker &mt) {
        return py::make_tuple(mt.mass_deltas(), mt.tolerance_ppm(),
                              mt.mass_mode(), mt.strict_count(),
                              mt.track_applications(),                              
                              mt.get_massmap(), mt.get_seen(), mt.counts(),
			      mt.applications());
          },
          [](py::tuple t) {
            if (t.size() != 9)
              throw std::runtime_error("Invalid state");

            ArrayType mass_deltas(t[0].cast<ConstArrayRef>());
            double tol_ppm(t[1].cast<double>());
            TransformMassMode mode(t[2].cast<TransformMassMode>());
            bool strict(t[3].cast<bool>());
            bool track_apps(t[4].cast<bool>());

            MassTracker result(mass_deltas, tol_ppm, mode, strict,
                                       track_apps);            
            result.set_state(t[5].cast<std::map<double, size_t>>(),
                             t[6].cast<std::vector<std::pair<size_t, int>>>(),
                             t[7].cast<IntArrayType>(),
                             t[8].cast<MassTracker::applications_t>());
	    return result;
            
	  }
	       )
	  );
  m.def("track_transforms", &track_transforms,
        "Find all transformations occurring in a list of masses.",
	"masses"_a, "mass_deltas"_a, "err_ppm"_a, "err_mode"_a);
  


  
  // Metrics
  m.def("compute_aff", &compute_aff,
      "Compute the affinity matrix from a similarity matrix",
      py::arg("sim"));
  m.def("jaccard", py::overload_cast<const Eigen::Ref<const BoolMatType>&>(&jaccard),
      "Compute the Jaccard similarity of a presence/absence matrix",
      py::arg("presence"));
  m.def("braycurtis", &braycurtis,
      "Compute the BrayCurtis similarity of an intensity matrix",
      py::arg("mat"));

  // Utils
  m.def("find_transforms", &find_transforms, "", py::arg("masses"),
        py::arg("xfrm_masses"), py::arg("err_abs"),
        py::arg("allow_overlap") = true);
  m.def("diffsort", &diffsort, "", py::arg("masses"));
  m.def("simplediffsort", &simplediffsort, "", py::arg("masses"));

  


}
