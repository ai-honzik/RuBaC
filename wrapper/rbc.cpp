#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../src/logger.cpp"
#include "../src/utils.cpp"
#include "../src/ruleset.cpp"
#include "../src/rule_learner.cpp"

namespace py = pybind11;

// TODO https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time

template <class CRuleLearnerBase = CRuleLearner>
class PyCRuleLearner : public CRuleLearnerBase{

  public:
    using CRuleLearnerBase::CRuleLearnerBase; // Inherit constructors
    CRuleset fit( const std::vector<std::vector<double>> & X,
                  const std::vector<std::size_t> & Y,
                  const std::vector<std::string> & feature_names,
                  std::size_t positive_class ) override {
      PYBIND11_OVERRIDE_PURE( CRuleset, CRuleLearnerBase, fit, X, Y, feature_names, positive_class );
    }
    std::vector<std::size_t> predict( const CRuleset & ruleset,
                                      const std::vector<std::vector<double>> & X,
                                      std::size_t positive_class ) const override{
      PYBIND11_OVERRIDE( std::vector<std::size_t>, CRuleLearnerBase, predict, ruleset, X, positive_class );
    }
};

PYBIND11_MODULE( rbc, m ){
  py::class_<CCondition>( m, "CCondition" )
    .def(py::init<const std::string &, std::size_t,
                  const std::string &, double>())
    .def(py::init<const std::string &, std::size_t,
                  const std::string &, const std::vector<double> &>())
    .def(py::init<const CCondition &>())
    .def("get_feature", &CCondition::get_feature)
    .def("get_index", &CCondition::get_index)
    .def("get_operator", &CCondition::get_operator)
    .def("get_values", &CCondition::get_values)
    .def("modify", static_cast<bool (CCondition::*)(const std::string &, double )>(&CCondition::modify))
    .def("modify", static_cast<bool (CCondition::*)(const CCondition &)>(&CCondition::modify))
    .def("covered_indices", &CCondition::covered_indices)
    .def("not_covered_indices", &CCondition::not_covered_indices)
    .def("to_string", &CCondition::to_string)
    .def("__copy__", []( const CCondition & self ){ return CCondition( self ); })
    .def("__str__", &CCondition::to_string)
    .def("__eq__", &CCondition::operator==)
    .def(py::pickle(
      []( const CCondition & cond ){ // __getstate__
        return py::make_tuple(
          cond.get_feature(),
          cond.get_index(),
          cond.get_operator(),
          cond.get_values() 
        );
      },
      []( py::tuple t ){ // __setstate__
        if( t.size() != 4 )
          throw std::runtime_error("Invalid condition tuple state!");
        CCondition cond(
          t[0].cast<std::string>(),
          t[1].cast<std::size_t>(),
          t[2].cast<std::string>(),
          t[3].cast<std::vector<double>>()
        );

        return cond;
      })
    ); 

  py::class_<CRule>( m, "CRule" )
    .def(py::init<>())
    .def(py::init<std::size_t, bool>())
    .def(py::init<const CRule &>())
    .def("add_cond", &CRule::add_cond)
    .def("predicted_class", &CRule::predicted_class)
    .def("predicts_class", &CRule::predicts_class)
    .def("predicts_the_same", &CRule::predicts_the_same)
    .def("to_string", &CRule::to_string)
    .def("size", &CRule::size)
    .def("covered_indices", &CRule::covered_indices)
    .def("not_covered_indices", &CRule::not_covered_indices)
    .def("learned_order", &CRule::learned_order)
    .def("__str__", &CRule::to_string)
    .def("__eq__", &CRule::operator==)
    .def("__setitem__", [](CRule & self, std::size_t i, const CCondition & value){ self[i] = value; })
    .def("__getitem__", static_cast<const CCondition & (CRule::*)(std::size_t) const>(&CRule::operator[]))
    .def("__copy__", []( const CRule & self ){ return CRule( self ); })
    .def("__len__", &CRule::size)
    .def(py::pickle(
      []( const CRule & rule ){ // __getstate__
        return py::make_tuple(
          rule.__pickle_get_cond(),
          rule.__pickle_get_learn_order(),
          rule.__pickle_get_class(),
          rule.__pickle_get_predict(),
          rule.__pickle_get_show_class()
        );
      },
      []( py::tuple t ){ // __setstate__
        if( t.size() != 5 )
          throw std::runtime_error("Invalid rule tuple state!");

        CRule rule;
        rule.__pickle_set_cond( t[0].cast<std::map<std::size_t,CCondition>>() );
        rule.__pickle_set_learn_order( t[1].cast<std::list<std::size_t>>() );
        rule.__pickle_set_class( t[2].cast<std::size_t>() );
        rule.__pickle_set_predict( t[3].cast<bool>() );
        rule.__pickle_set_show_class( t[4].cast<bool>() );

        return rule;
      })
    );

  py::class_<CRuleset>( m, "CRuleset" )
    .def(py::init<>())
    .def(py::init<const CRuleset &>())
    .def("add_rule", &CRuleset::add_rule)
    .def("pop", &CRuleset::pop)
    .def("to_string", &CRuleset::to_string)
    .def("size", &CRuleset::size)
    .def("covered_indices", &CRuleset::covered_indices)
    .def("not_covered_indices", &CRuleset::not_covered_indices)
    .def("covered_counts", &CRuleset::covered_counts)
    .def("__str__", &CRuleset::to_string)
    .def("__setitem__", [](CRuleset & self, std::size_t i, const CRule & value){ self[i] = value; })
    .def("__getitem__", static_cast<const CRule & (CRuleset::*)(std::size_t) const>(&CRuleset::operator[]))
    .def("__copy__", []( const CRuleset & self ){ return CRuleset( self ); })
    .def("__len__", &CRuleset::size)
    .def(py::pickle(
      []( const CRuleset & ruleset ){ // __getstate__
        return py::make_tuple(
          ruleset.__pickle_get_rules()
        );
      },
      []( py::tuple t ){
        if( t.size() != 1 )
          throw std::runtime_error("Invalid ruleset tuple state!");

        CRuleset ruleset;
        ruleset.__pickle_set_rules( t[0].cast<std::vector<CRule>>() );

        return ruleset;
      })
    );

  py::class_<CRuleLearner, PyCRuleLearner<>>( m, "CRuleLearner" )
    .def(py::init<>())
    .def(py::init<double, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                  bool, std::size_t, const std::string &>())
    // references won't get written to Python, but we can use sklearn there
    //.def_static("confusion_matrix", &CRuleLearner::confusion_matrix )
    .def_static("measure_accuracy", static_cast<double (*)(const std::vector<std::size_t> &, const std::vector<std::size_t> &)>(&CRuleLearner::measure_accuracy))
    .def_static("measure_accuracy", static_cast<double (*)(std::size_t, std::size_t, std::size_t, std::size_t)>(&CRuleLearner::measure_accuracy))
    .def("fit", &CRuleLearner::fit)
    // references not working
    //.def("pos_neg_split", &CRuleLearner::pos_neg_split)
    // reference not working
    //.def("data_split", &CRuleLearner::data_split)
    .def("grow_rule", static_cast<CRule (CRuleLearner::*)(const std::vector<std::vector<double>> &,
                                                          const std::vector<std::string> &,
                                                          const std::vector<std::size_t> &,
                                                          const std::vector<std::size_t> &)>(&CRuleLearner::grow_rule))
    .def("grow_rule", static_cast<CRule (CRuleLearner::*)(const std::vector<std::vector<double>> &,
                                                          const std::vector<std::string> &,
                                                          const std::vector<std::size_t> &,
                                                          const std::vector<std::size_t> &,
                                                          const CRule & r)>(&CRuleLearner::grow_rule))
    .def("find_literal", &CRuleLearner::find_literal)
    //.def("foil_metric", &CRuleLearner::foil_metric)
    .def("prune_rule", &CRuleLearner::prune_rule)
    //.def("IREP_pruning_metric", &CRuleLearner::IREP_pruning_metric)
    .def("rule_error", &CRuleLearner::rule_error)
    .def("predict", &CRuleLearner::predict);

  py::class_<COneR, CRuleLearner, PyCRuleLearner<COneR>>( m, "COneR" )
    .def(py::init<>())
    .def("fit", &COneR::fit)
    .def("predict", static_cast<std::vector<std::size_t> (COneR::*)(const CRuleset &,
                                                                    const std::vector<std::vector<double>> &,
                                                                    std::size_t) const>(&COneR::predict),
         py::arg("ruleset"), py::arg("X"), py::arg("positive_class") = 0 );

  py::class_<CIREP, CRuleLearner, PyCRuleLearner<CIREP>>( m, "CIREP" )
    .def(py::init<>())
    .def(py::init<double, std::size_t, std::size_t, std::size_t, std::size_t,
                  bool, std::size_t, const std::string &>(),
         py::arg("split_ratio") = (double)2/3, py::arg("random_state") = std::random_device()(), py::arg("categorical_max") = 0,
         py::arg("rule_size") = std::numeric_limits<std::size_t>::max(),
         py::arg("ruleset_size") = std::numeric_limits<std::size_t>::max(),
         py::arg("prune_rules") = true, py::arg("n_threads") = 1, py::arg("pruning_metric") = "IREP_default" )
    .def("fit", &CIREP::fit);

  py::class_<CRIPPER, CRuleLearner, PyCRuleLearner<CRIPPER>>( m, "CRIPPER" )
    .def(py::init<>())
    .def(py::init<double, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                  std::size_t, bool, std::size_t, const std::string &>(),
         py::arg("split_ratio") = (double)2/3, py::arg("random_state") = std::random_device()(), py::arg("categorical_max") = 0,
         py::arg("difference") = 64, py::arg("rule_size") = std::numeric_limits<std::size_t>::max(),
         py::arg("ruleset_size") = std::numeric_limits<std::size_t>::max(),
         py::arg("k") = 2, py::arg("prune_rules") = true, py::arg("n_threads") = 1,
         py::arg("pruning_metric") = "RIPPER_default" )
    .def("fit", &CRIPPER::fit)
    .def("optimise_ruleset", &CRIPPER::optimise_ruleset);

  py::class_<CCompetitor, CRuleLearner, PyCRuleLearner<CCompetitor>>( m, "CCompetitor" )
    .def(py::init<>())
    .def(py::init<double, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                  bool, std::size_t, const std::string &>(),
         py::arg("split_ratio") = (double)2/3, py::arg("random_state") = std::random_device()(), py::arg("categorical_max") = 0,
         py::arg("rule_size") = std::numeric_limits<std::size_t>::max(),
         py::arg("ruleset_size") = std::numeric_limits<std::size_t>::max(),
         py::arg("difference") = 64, py::arg("prune_rules") = true, py::arg("n_threads") = 1,
         py::arg("pruning_metric") = "RIPPER_default" )
    .def("fit", &CCompetitor::fit);
}
