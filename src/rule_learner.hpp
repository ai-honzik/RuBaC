#ifndef __rule_learnerhpp__
#define __rule_learnerhpp__

#include <vector>
#include <map>
#include <set>
#include <numeric>
#include <algorithm>
#include <limits>
#include <random>
#include <cmath>
#include <iterator>
#include <functional>
#include "./ruleset.hpp"
#include "./utils.hpp"

#ifdef __verbose__
  #include "logger.hpp"
  extern CLogger __logger;
#endif

class CRuleLearner{

  public:
    CRuleLearner( void );
    CRuleLearner( double split_ratio, std::size_t random_state, 
                  std::size_t categorical_max, std::size_t difference,
                  bool prune_rules, std::size_t n_threads,
                  const std::string & pruning_metric );
    static void confusion_matrix( const std::vector<std::size_t> & y_true,
                                  const std::vector<std::size_t> & y_pred,
                                  std::size_t & tn, std::size_t & fp,
                                  std::size_t & fn, std::size_t & tp );
    static void confusion_matrix( const CRuleset & ruleset,
                                  std::size_t start_index,
                                  const std::vector<std::vector<double>> & X,
                                  const std::vector<std::size_t> & pos,
                                  const std::vector<std::size_t> & neg, 
                                  std::size_t & tn, std::size_t & fp,
                                  std::size_t & fn, std::size_t & tp );
    static double measure_accuracy( const std::vector<std::size_t> & y_true,
                                    const std::vector<std::size_t> & y_pred );
    static double measure_accuracy( std::size_t tn, std::size_t fp,
                                    std::size_t fn, std::size_t tp );
    virtual CRuleset fit( const std::vector<std::vector<double>> & X,
                          const std::vector<std::size_t> & Y,
                          const std::vector<std::string> & feature_names,
                          std::size_t positive_class ) = 0;
    // division between positive and negative indices
    void pos_neg_split( const std::vector<std::size_t> & Y,
                        std::size_t positive_class,
                        std::vector<std::size_t> & pos,
                        std::vector<std::size_t> & neg ) const;
    // data split
    void data_split( const std::vector<std::size_t> & input_indices,
                     std::vector<std::size_t> & a,
                     std::vector<std::size_t> & b );
    // grow rule
    CRule grow_rule( const std::vector<std::vector<double>> & X,
                     const std::vector<std::string> & feature_names,
                     const std::vector<std::size_t> & pos_grow,
                     const std::vector<std::size_t> & neg_grow );

    CRule grow_rule( const std::vector<std::vector<double>> & X,
                     const std::vector<std::string> & feature_names,
                     const std::vector<std::size_t> & pos_grow,
                     const std::vector<std::size_t> & neg_grow,
                     const CRule & r );
    CCondition * find_literal( const std::vector<std::vector<double>> & X,
                               const std::vector<std::string> & feature_names,
                               const std::vector<std::size_t> & pos_grow,
                               const std::vector<std::size_t> & neg_grow,
                               std::size_t pos_size, std::size_t neg_size );
    void foil_metric( const std::map<double,std::size_t> & pos_sums,
                      const std::map<double,std::size_t> & neg_sums,
                      std::size_t pos_size, std::size_t neg_size,
                      const std::string & feature, std::size_t index,
                      const std::string & m_op, double & best_gain,
                      CCondition *& best_cond ) const;
    CRule prune_rule( const CRule & old_rule,
                      const std::vector<std::vector<double>> & X,
                      const std::vector<std::size_t> & pos_prune,
                      const std::vector<std::size_t> & neg_prune );
    double rule_error( const std::vector<std::vector<double>> & X,
                       const CRule & rule,
                       const std::vector<std::size_t> & pos_prune,
                       const std::vector<std::size_t> & neg_prune ) const;
    virtual std::vector<std::size_t> predict(
                        const CRuleset & ruleset,
                        const std::vector<std::vector<double>> & X,
                        std::size_t positive_class ) const;
    void set_pruning_metric( const std::string & metric );
    double total_description_length( const CRuleset & ruleset,
                                     const std::vector<std::vector<double>> & X,
                                     const std::vector<std::size_t> & y_true,
                                     std::size_t positive_class ) const;
    double total_description_length( const CRuleset & ruleset,
                                     const std::vector<std::vector<double>> & X,
                                     const std::vector<std::size_t> & y_true,
                                     std::size_t positive_class,
                                     std::size_t conditions_count ) const;
    double total_description_length( const std::vector<std::vector<double>> & X,
                                     const CRuleset & new_ruleset,
                                     const CRuleset & old_ruleset,
                                     std::size_t rule_index,
                                     const std::vector<std::size_t> & pos,
                                     const std::vector<std::size_t> & neg,
                                     std::size_t tn, std::size_t fp,
                                     std::size_t fn, std::size_t tp,
                                     std::size_t & tn_r, std::size_t & fp_r,
                                     std::size_t & fn_r, std::size_t & tp_r,
                                     double RDL, double & RDL_r,
                                     std::size_t conditions_count ) const;
    double rule_bits( const CRule & rule, std::size_t conditions_count ) const;
    double exception_bits( const CRuleset & ruleset,
                           const std::vector<std::vector<double>> & X,
                           const std::vector<std::size_t> & y_true,
                           std::size_t positive_class ) const;
    double exception_bits( std::size_t tn, std::size_t fp,
                           std::size_t fn, std::size_t tp ) const;
    std::size_t unique_conditions( const std::vector<std::vector<double>> & X ) const;
    std::size_t ruleset_coverage_diff( const std::vector<std::vector<double>> & X, 
                                       const CRuleset & ruleset,
                                       const std::vector<std::size_t> & covered_a,
                                       const std::vector<std::size_t> & covered_b ) const;

  protected:
    double m_split_ratio; // split ratio for current learner
    std::size_t m_random_state; // random state for init. of m_rand_gen
    std::size_t m_categorical_max; // maximum number of unique vals in a feature
    std::size_t m_difference;
    bool m_prune_rules; // should rules be pruned?
    std::size_t m_n_threads;
    std::function<double( const std::vector<std::vector<double>> & X,
                          const CRule & rule,
                          const std::vector<std::size_t> & pos_prune,
                          const std::vector<std::size_t> & neg_prune )> m_pruning_metric;
    std::mt19937_64 m_rand_gen;
};

class CIREP : public CRuleLearner{

  public:
    CIREP( void );
    CIREP( double split_ratio, std::size_t random_state=std::random_device()(), 
           std::size_t categorical_max=0, bool prune_rules=true,
           std::size_t n_threads=1,
           const std::string & pruning_metric="IREP_default" );
    virtual CRuleset fit( const std::vector<std::vector<double>> & X,
                          const std::vector<std::size_t> & Y,
                          const std::vector<std::string> & feature_names,
                          std::size_t positive_class );
};

class CRIPPER : public CRuleLearner{

  public:
    CRIPPER( void );
    CRIPPER( double split_ratio, std::size_t random_state=std::random_device()(),
             std::size_t categorical_max=0, std::size_t difference=64,
             std::size_t k=2, bool prune_rules=true, std::size_t n_threads=1, 
             const std::string & pruning_metric="RIPPER_default" );
    CRuleset IREP_star( const std::vector<std::vector<double>> & X,
                        const std::vector<std::size_t> & Y,
                        const std::vector<std::size_t> & pos,
                        const std::vector<std::size_t> & neg,
                        const std::vector<std::string> & feature_names,
                        std::size_t positive_class,
                        const CRuleset & input_ruleset );
    virtual CRuleset fit( const std::vector<std::vector<double>> & X,
                          const std::vector<std::size_t> & Y,
                          const std::vector<std::string> & feature_names,
                          std::size_t positive_class );
    CRuleset optimise_ruleset( const CRuleset & input_ruleset,
                               const std::vector<std::vector<double>> & X,
                               const std::vector<std::string> & feature_names,
                               const std::vector<std::size_t> & pos,
                               const std::vector<std::size_t> & neg );
    CRule optimise_prune( const CRuleset & input_ruleset,
                          std::size_t index,
                          const std::vector<std::vector<double>> & X,
                          const std::vector<std::size_t> & pos_prune,
                          const std::vector<std::size_t> & neg_prune );
    CRuleset generalise_ruleset( const CRuleset & input_ruleset,
                                 const std::vector<std::vector<double>> & X, 
                                 const std::vector<std::size_t> & Y,
                                 std::size_t positive_class ) const;

  private:
    std::size_t m_k;

};

class CCompetitor : public CRuleLearner{

  public:
    CCompetitor( void );
    CCompetitor( double split_ratio, std::size_t random_state=std::random_device()(),
                 std::size_t categorical_max=0, std::size_t difference=64,
                 bool prune_rules=true, std::size_t n_threads=1,
                 const std::string & pruning_metric="RIPPER_default" );
    virtual CRuleset fit( const std::vector<std::vector<double>> & X,
                          const std::vector<std::size_t> & Y,
                          const std::vector<std::string> & feature_names,
                          std::size_t positive_class );
};

class COneR : public CRuleLearner{

  public:
    COneR( void );
    virtual CRuleset fit( const std::vector<std::vector<double>> & X,
                          const std::vector<std::size_t> & Y,
                          const std::vector<std::string> & feature_names,
                          std::size_t positive_class );
    virtual std::vector<std::size_t> predict( 
                        const CRuleset & ruleset,
                        const std::vector<std::vector<double>> & X,
                        std::size_t positive_class ) const;
    virtual std::vector<std::size_t> predict( 
                        const CRuleset & ruleset,
                        const std::vector<std::vector<double>> & X ) const;

  private:
    // TODO
    // min classes?
    // randomness?
    // categorical max?

    CRuleset discretise( std::size_t row,
                         const std::vector<std::vector<double>> & X,
                         const std::vector<std::size_t> & Y,
                         const std::vector<std::string> & feature_names,
                         std::size_t positive_class,
                         std::size_t min_class=3 ) const;
    CRuleset simplify_ruleset( const CRuleset & ruleset,
                               std::size_t row ) const;
};

#endif /*__rule_learnerhpp__*/
