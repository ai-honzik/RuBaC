#ifndef __rule_learnercpp__
#define __rule_learnercpp__

#include "./rule_learner.hpp"

CRuleLearner::CRuleLearner( void ):
    m_split_ratio( 2./3 ), m_categorical_max( 0 ), m_difference( 64 ),
    m_prune_rules( true ), m_n_threads( 1 ){

  std::random_device rand_dev;
  m_random_state = rand_dev();
  m_rand_gen = std::mt19937_64( m_random_state );

}

CRuleLearner::CRuleLearner( double split_ratio, std::size_t random_state,
                            std::size_t categorical_max, std::size_t difference,
                            bool prune_rules, std::size_t n_threads,
                            const std::string & pruning_metric ):
    m_split_ratio( split_ratio ), m_random_state( random_state ),
    m_categorical_max( categorical_max ), m_difference( difference ),
    m_prune_rules( prune_rules ), m_n_threads( n_threads ),
    m_rand_gen( random_state ){
  set_pruning_metric( pruning_metric );
}

void CRuleLearner::confusion_matrix( const std::vector<std::size_t> & y_true,
                                     const std::vector<std::size_t> & y_pred,
                                     std::size_t & tn, std::size_t & fp,
                                     std::size_t & fn, std::size_t & tp ){

  if( y_true.size() != y_pred.size() )
    throw std::invalid_argument( "Input vector sizes differ!" );

  tn = fp = fn = tp = 0;

  for( std::size_t i = 0; i < y_true.size(); ++i ){
    if( y_true[i] && y_true[i] == y_pred[i] ) ++tp;
    else if( ! y_true[i] && y_true[i] == y_pred[i] ) ++tn;
    else if( y_true[i] && ! y_pred[i] ) ++fn;
    else if( ! y_true[i] && y_pred[i] ) ++fp;
    else
      throw std::runtime_error( "Wrong option in confusion matrix!" );
  }
}

void CRuleLearner::confusion_matrix( const CRuleset & ruleset,
                                     std::size_t start_index,
                                     const std::vector<std::vector<double>> & X,
                                     const std::vector<std::size_t> & pos,
                                     const std::vector<std::size_t> & neg, 
                                     std::size_t & tn, std::size_t & fp,
                                     std::size_t & fn, std::size_t & tp ){

  if( ! start_index && ! ruleset.size() ){
    tp = 0;
    fn = pos.size(); // empty ruleset does not cover anything
    fp = 0;
    tn = neg.size(); // negative samples are not covered, thus they're
                     // true negative
    return;
  }
  else if( start_index >= ruleset.size() )
    throw std::invalid_argument( "Ruleset index out of range!" );

  std::vector<std::size_t> pos_copy( pos );
  std::vector<std::size_t> neg_copy( neg );

  tp = pos_copy.size();
  fp = neg_copy.size();
  fn = tn = 0;

  for( std::size_t i = start_index; i < ruleset.size(); ++i ){
    const auto & rule = ruleset[i];
    pos_copy = rule.not_covered_indices( X, pos_copy );
    neg_copy = rule.not_covered_indices( X, neg_copy );
  }

  tp -= pos_copy.size();
  fn = pos_copy.size();
  fp -= neg_copy.size();
  tn = neg_copy.size();
}

double CRuleLearner::measure_accuracy( const std::vector<std::size_t> & y_true,
                                       const std::vector<std::size_t> & y_pred ){
  std::size_t tn, fp, fn, tp;
  confusion_matrix( y_true, y_pred, tn, fp, fn, tp );
  #ifdef __verbose__
    __logger.log( "(tn, fp, fn, tp) = (" + std::to_string( tn ) + ", " +
                  std::to_string( fp ) + ", " + std::to_string( fn ) +
                  ", " + std::to_string( tp ) + ")" );
  #endif

  return measure_accuracy( tn, fp, fn, tp );
}

double CRuleLearner::measure_accuracy( std::size_t tn, std::size_t fp,
                                       std::size_t fn, std::size_t tp ){
  return (double)( tp + tn ) / ( tp + tn + fp + fn );
}

void CRuleLearner::pos_neg_split( const std::vector<std::size_t> & Y,
                                  std::size_t positive_class,
                                  std::vector<std::size_t> & pos,
                                  std::vector<std::size_t> & neg ) const{

  for( std::size_t i = 0; i < Y.size(); ++i ){
    if( Y[i] == positive_class )
      pos.push_back( i );
    else
      neg.push_back( i );
  }

}

void CRuleLearner::data_split( const std::vector<std::size_t> & input_indices,
                               std::vector<std::size_t> & a,
                               std::vector<std::size_t> & b ){ 

  if( ! input_indices.size() )
    throw std::invalid_argument( "Empty input vector!" );

  std::vector<std::size_t> indices( input_indices );
  std::shuffle( indices.begin(), indices.end(), m_rand_gen );
  std::size_t split_val = (std::size_t)std::ceil( m_split_ratio * input_indices.size() );

  if( split_val > input_indices.size() )
    throw std::runtime_error( "Split underflow!" );
  else if( ! split_val )
    throw std::runtime_error( "Split value is 0!" );
  // TODO should throw when b.size() == 0 ?

  auto it = indices.begin();
  std::advance( it, split_val );
  a = std::vector<std::size_t>( std::make_move_iterator( indices.begin() ),
                                std::make_move_iterator( it ) );
  b = std::vector<std::size_t>( std::make_move_iterator( it ),
                                std::make_move_iterator( indices.end() ) );

  // indices need to be kept in sorted order
  std::sort( a.begin(), a.end() );
  std::sort( b.begin(), b.end() );
}

CRule CRuleLearner::grow_rule( const std::vector<std::vector<double>> & X,
                               const std::vector<std::string> & feature_names,
                               const std::vector<std::size_t> & pos_grow,
                               const std::vector<std::size_t> & neg_grow ){

  CRule r;
  r = grow_rule( X, feature_names, pos_grow, neg_grow, r );
  return r;
}

CRule CRuleLearner::grow_rule( const std::vector<std::vector<double>> & X,
                               const std::vector<std::string> & feature_names,
                               const std::vector<std::size_t> & pos_grow,
                               const std::vector<std::size_t> & neg_grow,
                               const CRule & r ){

  CRule rule( r );

  std::vector<std::size_t> pos_copy( pos_grow );
  std::vector<std::size_t> neg_copy( neg_grow );

  if( rule.size() ){
    pos_copy = rule.covered_indices( X, pos_grow );
    neg_copy = rule.covered_indices( X, neg_grow );
  }

  while( ! neg_copy.empty() ){

    CRule old_rule( rule );
    CCondition * cond = find_literal( X, feature_names, pos_copy, neg_copy,
                                      pos_copy.size(), neg_copy.size() );
    // check if the condition is not empty
    if( ! cond ){
      #ifdef __verbose__
        __logger.log( "---- No better condition could have been found." );
      #endif
      break;
    }

    #ifdef __verbose__
      __logger.log( "---- Found condition: " + (*cond).to_string() );
    #endif

    // add condition to the rule and delete it
    rule.add_cond( CCondition( *cond ) );
    delete cond;

    // check whether the old_rule and rule do not match
    if( old_rule == rule ){
      #ifdef __verbose__
        __logger.log( "---- Generated the same rule with length: " +
                      std::to_string( rule.size() ) );
      #endif
      break;
    }

    // change pos_copy and neg_copy to covered samples
    pos_copy = rule.covered_indices( X, pos_copy );
    neg_copy = rule.covered_indices( X, neg_copy );
  }

  #ifdef __verbose__
    if( rule.size() < 1 )
      __logger.log( "---- Rule has no conditions!" );
  #endif

  return rule;
}

CCondition * CRuleLearner::find_literal( const std::vector<std::vector<double>> & X,
                                         const std::vector<std::string> & feature_names,
                                         const std::vector<std::size_t> & pos_grow,
                                         const std::vector<std::size_t> & neg_grow,
                                         std::size_t pos_size, std::size_t neg_size ){
  CCondition * best_cond = nullptr;
  double best_gain = std::numeric_limits<double>::lowest();

  for( std::size_t i = 0; i < feature_names.size(); ++i ){
  
    auto & X_row = X[i];

    auto pos_uniq = unique_counts( X_row, pos_grow );
    auto neg_uniq = unique_counts( X_row, neg_grow );
    std::map<double,std::size_t> pos_sums; 
    std::map<double,std::size_t> neg_sums; 
    std::string used_op;

    std::vector<std::string> ops;
    if( m_categorical_max && pos_uniq.size() <= m_categorical_max )
      ops.push_back( "in" );
    else{
      ops.push_back( "<=" );
      ops.push_back( ">=" );
    }

    for( auto & op : ops ){
      if( op == "in" ){
        used_op = "in";
        pos_sums = std::map<double,std::size_t>( std::move( pos_uniq ) ); 
        for( auto & it : pos_sums ){
          auto searched_uniq = neg_uniq.find( it.first );
          if( searched_uniq != neg_uniq.end() )
            neg_sums.insert( { it.first, searched_uniq -> second } );
          else
            neg_sums.insert( { it.first, 0 } );
          // neg_sums[ it.first ] = neg_uniq[ it.first ];
        }
      }
      else if( op == "<=" ){
        used_op = "<=";
        pos_sums = std::map<double,std::size_t>( std::move( pos_uniq ) );
        map_cum_sum_ip( pos_sums.begin(), pos_sums.end() );
        map_cum_sum_ip( neg_uniq.begin(), neg_uniq.end() );

        for( auto & it : pos_sums ){
          auto searched_uniq = neg_uniq.upper_bound( it.first );
          if( searched_uniq == neg_uniq.begin() )
            // if searched points to .begin(), neg_uniq does
            // not contain any values `<= x`.
            // neg_sums[ it.first ] = 0;
            neg_sums.insert( { it.first, 0 } );
          else{
            // the iterator needs to be decremented as the
            // value found (searched) is the first value
            // after the searched value
            --searched_uniq;
            // neg_sums[ it.first ] = searched_uniq -> second;
            neg_sums.insert( { it.first, searched_uniq -> second } );
          }
        }
      }
      else if( op == ">=" ){
        used_op = ">=";
        pos_sums = std::map<double,std::size_t>( std::move( pos_uniq ) );
        map_cum_sum_ip( pos_sums.rbegin(), pos_sums.rend() );
        map_cum_sum_ip( neg_uniq.rbegin(), neg_uniq.rend() );

        for( auto & it : pos_sums ){
          auto searched_uniq = neg_uniq.lower_bound( it.first );
          if( searched_uniq == neg_uniq.end() )
            // if we reached .end() then there are no values `>= x`.
            // neg_sums[ it.first ] = 0;
            neg_sums.insert( { it.first, 0 } );
          else
            // neg_sums[ it.first ] = searched -> second;
            neg_sums.insert( { it.first, searched_uniq -> second } );
        }
      }
      // proceed if both sums are non-empty
      if( ! pos_sums.empty() && ! neg_sums.empty() )
        foil_metric( pos_sums, neg_sums, pos_size, neg_size, feature_names[i],
                     i, used_op, best_gain, best_cond );
    }
  }

  return best_cond;

}

void CRuleLearner::foil_metric( const std::map<double,std::size_t> & pos_sums,
                                const std::map<double,std::size_t> & neg_sums,
                                std::size_t pos_size, std::size_t neg_size,
                                const std::string & feature, std::size_t index,
                                const std::string & m_op, double & best_gain,
                                CCondition *& best_cond ) const{
  auto it_pos = pos_sums.begin();
  auto it_neg = neg_sums.begin();

  double old_log = std::log( (double)pos_size / ( pos_size + neg_size ) );

  for( ; it_pos != pos_sums.end() && it_neg != neg_sums.end(); ++it_pos, ++it_neg ){
    auto & pos = it_pos -> second;
    auto & neg = it_neg -> second;
    double new_log = std::log( (double) pos / ( pos + neg ) );
    double foil = pos * ( new_log - old_log );

    if( foil > best_gain ){
      best_gain = foil;
      
      if( best_cond )
        delete best_cond;
      best_cond = new CCondition( feature, index, m_op, it_pos -> first );
    }
  }
}

CRule CRuleLearner::prune_rule( const CRule & old_rule,
                                const std::vector<std::vector<double>> & X,
                                const std::vector<std::size_t> & pos_prune,
                                const std::vector<std::size_t> & neg_prune ){
  double best_val = m_pruning_metric( X, old_rule, pos_prune, neg_prune );
  auto learned_order = old_rule.learned_order();
  CRule r( old_rule );

  for( auto it = old_rule.o_crbegin(); it != old_rule.o_crend(); ++it ){
    CRule new_rule( r );
    new_rule.pop_back();

    double new_val = m_pruning_metric( X, new_rule, pos_prune, neg_prune );
    #ifdef __verbose__
      __logger.log( "---- Old acc: " + std::to_string( best_val ) +
                    ", new acc: " + std::to_string( new_val ) +
                    ", pos_prune size: " + std::to_string( pos_prune.size() ) +
                    ", neg_prune size: " + std::to_string( neg_prune.size() ) );
    #endif
    if( new_rule.size() && new_val > best_val ){
      best_val = new_val;
      r = new_rule;
    }
    else
      break;

  }

  return r;

}

double CRuleLearner::rule_error( const std::vector<std::vector<double>> & X,
                                 const CRule & rule,
                                 const std::vector<std::size_t> & pos_prune,
                                 const std::vector<std::size_t> & neg_prune ) const{
  double p,n;
  p = rule.covered_indices( X, pos_prune ).size();
  n = rule.covered_indices( X, neg_prune ).size();

  if( n < 1 && p < 1 )
    return 0;

  return p/(p+n);
}

std::vector<std::size_t> CRuleLearner::predict(
                        const CRuleset & ruleset,
                        const std::vector<std::vector<double>> & X,
                        std::size_t positive_class ) const{

  if( ! X.size() )
    throw std::invalid_argument( "Empty data!" );

  std::vector<std::size_t> predicted( X.front().size(), 0 );
  std::vector<std::size_t> indices( X.front().size() );
  std::iota( indices.begin(), indices.end(), 0 );

  for( std::size_t i = 0; i < ruleset.size(); ++i ){
    auto covered = ruleset[i].covered_indices( X, indices ); 
    for( auto & j : covered )
      predicted[j] = positive_class;
    indices = ruleset[i].not_covered_indices( X, indices );
  }

  return predicted;
}

void CRuleLearner::set_pruning_metric( const std::string & metric ){
  if( metric == "IREP_default" )
    m_pruning_metric = IREP_pruning_metric;
  else if( metric == "RIPPER_default" )
    m_pruning_metric = RIPPER_pruning_metric;
  else
    throw std::runtime_error( "Invalid pruning metric!" );
}

double CRuleLearner::total_description_length( const CRuleset & ruleset,
                                               const std::vector<std::vector<double>> & X,
                                               const std::vector<std::size_t> & y_true,
                                               std::size_t positive_class ) const{
  std::size_t conditions_count = unique_conditions( X ); 
  return total_description_length( ruleset, X, y_true, positive_class, conditions_count );
}

double CRuleLearner::total_description_length( const CRuleset & ruleset,
                                               const std::vector<std::vector<double>> & X,
                                               const std::vector<std::size_t> & y_true,
                                               std::size_t positive_class,
                                               std::size_t conditions_count ) const{
  double DL = 0.;

  for( std::size_t i = 0; i < ruleset.size(); ++i )
    DL += rule_bits( ruleset[i], conditions_count );

  DL += exception_bits( ruleset, X, y_true, positive_class );

  return DL;
}

double CRuleLearner::total_description_length( const std::vector<std::vector<double>> & X,
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
                                               std::size_t conditions_count ) const{
  CRule r_new = new_ruleset[rule_index];
  CRule r_old = old_ruleset[rule_index]; 

  auto pos_covered_old = r_old.covered_indices( X, pos );
  auto neg_covered_old = r_old.covered_indices( X, neg );
  auto pos_covered_new = r_new.covered_indices( X, pos );
  auto neg_covered_new = r_new.covered_indices( X, neg );

  // init replacement vars
  tn_r = tn;
  fp_r = fp;
  fn_r = fn;
  tp_r = tp;

  std::size_t diff;
  // TP and FN change
  // if `| old_cov \ new_cov | > 0` then
  //   diff = new_ruleset.cov( old_cov \ new_cov )
  diff = ruleset_coverage_diff( X, new_ruleset, pos_covered_old, pos_covered_new );
  tp_r -= diff;
  fn_r += diff;

  // if `| new_cov \ old_cov | > 0` then
  //   diff = old_ruleset.cov( new_cov \ old_cov )
  diff = ruleset_coverage_diff( X, old_ruleset, pos_covered_new, pos_covered_old );
  tp_r += diff;
  fn_r -= diff;  

  // TN and FP change
  // if `| old_cov \ new_cov | > 0` then
  //   diff = new_ruleset.cov( old_cov \ new_cov )
  diff = ruleset_coverage_diff( X, new_ruleset, neg_covered_old, neg_covered_new );
  fp_r -= diff;
  tn_r += diff;

  // if `| new_cov \ old_cov | > 0` then
  //   diff = old_ruleset.cov( new_cov \ old_cov )
  diff = ruleset_coverage_diff( X, old_ruleset, neg_covered_new, neg_covered_old );
  fp_r += diff;
  tn_r -= diff; 

  // find out how many bits had replaced rule and new rule
  double rule_bits_old = rule_bits( old_ruleset[rule_index], conditions_count );
  double rule_bits_new = rule_bits( new_ruleset[rule_index], conditions_count );
  RDL_r = RDL - rule_bits_old + rule_bits_new;

  // calculate exception bits with new vars
  double exceptions = exception_bits( tn_r, fp_r, fn_r, tp_r ); 

  // and finally return the result
  return RDL_r + exceptions;
}

double CRuleLearner::rule_bits( const CRule & rule, std::size_t conditions_count ) const{

  std::size_t k = rule.size();
  auto & n = conditions_count;
  double p = (double)k / n;
  double bits = 0.;

  // TODO edge cases!
  // calculate the number of bits given by the formula
  bits = ( k * std::log2( 1/p ) + ( n - k ) * std::log2( 1/( 1 - p ) ) +
           std::log2( k ) ) * 0.5;

  return bits;

}

double CRuleLearner::exception_bits( const CRuleset & ruleset,
                                     const std::vector<std::vector<double>> & X,
                                     const std::vector<std::size_t> & y_true,
                                     std::size_t positive_class ) const{
  auto predicted = predict( ruleset, X, positive_class );
  std::size_t tn, fp, fn, tp;
  confusion_matrix( y_true, predicted, tn, fp, fn, tp );

  return exception_bits( tn, fp, fn, tp );
}

double CRuleLearner::exception_bits( std::size_t tn, std::size_t fp,
                                     std::size_t fn, std::size_t tp ) const{
  return Slog_C( tp + fp, fp ) + Slog_C( tn + fn, fn );
}

std::size_t CRuleLearner::unique_conditions( const std::vector<std::vector<double>> & X ) const{

  std::size_t count = 0;

  for( const auto & row : X )
    count += unique( row ).size();

  return count; 
}

std::size_t CRuleLearner::ruleset_coverage_diff(
  const std::vector<std::vector<double>> & X, 
  const CRuleset & ruleset,
  const std::vector<std::size_t> & covered_a,
  const std::vector<std::size_t> & covered_b
) const{
  std::vector<std::size_t> diff;

  std::set_difference( covered_a.begin(), covered_a.end(),
                       covered_b.begin(), covered_b.end(),
                       std::inserter( diff, diff.begin() ) );
  // no difference
  if( diff.empty() )
    return 0;
  
  auto not_covered_rest = ruleset.not_covered_indices( X, diff );
  return not_covered_rest.size();
   
}

CIREP::CIREP( void ):
    CRuleLearner(){
  set_pruning_metric( "IREP_default" );
}

CIREP::CIREP( double split_ratio, std::size_t random_state, 
              std::size_t categorical_max, bool prune_rules,
              std::size_t n_threads,
              const std::string & pruning_metric ):
    CRuleLearner( split_ratio, random_state, categorical_max,
                  64, prune_rules, n_threads, pruning_metric ){
}

CRuleset CIREP::fit( const std::vector<std::vector<double>> & X,
                     const std::vector<std::size_t> & Y,
                     const std::vector<std::string> & feature_names,
                     std::size_t positive_class ){

  if( ! X.size() || ! Y.size() )
    throw std::invalid_argument( "Input vectors are empty!" );
  else if( X.front().size() != Y.size() )
    throw std::invalid_argument( "X and Y sizes differ!" );
  else if( X.size() != feature_names.size() )
    throw std::invalid_argument( "Y and feature names differ!" );

  std::vector<std::size_t> pos;
  std::vector<std::size_t> neg;
  pos_neg_split( Y, positive_class, pos, neg );

  CRuleset ruleset;
  std::vector<std::size_t> pos_grow,pos_prune;
  std::vector<std::size_t> neg_grow,neg_prune;

  while( ! pos.empty() ){    

    #ifdef __verbose__
      __logger.log( "Pos: " + std::to_string( pos.size() ) + ", Neg: " +
                    std::to_string( neg.size() ) + " remaining." );
    #endif

    data_split( pos, pos_grow, pos_prune );
    data_split( neg, neg_grow, neg_prune );

    // grow a rule
    #ifdef __verbose__
      __logger.log( "-- Growing" );
    #endif
    auto rule = grow_rule( X, feature_names, pos_grow, neg_grow );

    // prune the rule
    if( m_prune_rules ){
      #ifdef __verbose__
        __logger.log( "-- Pruning rule with size: " + std::to_string( rule.size() ) );
      #endif
      rule = prune_rule( rule, X, pos_prune, neg_prune );
      #ifdef __verbose__
        __logger.log("-- Pruned rule has size: " + std::to_string( rule.size() ) );
        __logger.log( "-- Rule error: " + std::to_string(
                      rule_error( X, rule, pos_prune, neg_prune ) ) );
      #endif
    }
    if( rule_error( X, rule, pos_prune, neg_prune ) < 0.5 ){
      #ifdef __verbose__
        __logger.log( "-- Rule error > 50 %, stopping..." );
      #endif
      break;
    }

    pos = rule.not_covered_indices( X, pos );
    neg = rule.not_covered_indices( X, neg );
    // add to ruleset
    ruleset.add_rule( rule );

  }

  return ruleset;
}

CRIPPER::CRIPPER( void ):
    CRuleLearner(), m_k( 2 ){
}

CRIPPER::CRIPPER( double split_ratio, std::size_t random_state, 
                  std::size_t categorical_max, std::size_t difference,
                  std::size_t k, bool prune_rules, std::size_t n_threads,
                  const std::string & pruning_metric ):
    CRuleLearner( split_ratio, random_state, categorical_max,
                  difference, prune_rules, n_threads, pruning_metric ), m_k( k ){
}

CRuleset CRIPPER::IREP_star( const std::vector<std::vector<double>> & X,
                             const std::vector<std::size_t> & Y,
                             const std::vector<std::size_t> & pos,
                             const std::vector<std::size_t> & neg,
                             const std::vector<std::string> & feature_names,
                             std::size_t positive_class,
                             const CRuleset & input_ruleset ){

  if( pos.empty() ){
    #ifdef __verbose__
      __logger.log( "-- Cannot find rules for empty pos indices!" );
    #endif
    return input_ruleset;
  }

  CRuleset ruleset( input_ruleset );
  auto pos_copy = pos;
  auto neg_copy = neg;
  std::vector<std::size_t> pos_grow, pos_prune;
  std::vector<std::size_t> neg_grow, neg_prune;

  std::size_t tn, fp, fn, tp;
  tn = fp = fn = tp = 0;
  confusion_matrix( ruleset, 0, X, pos, neg, tn, fp, fn, tp );

  // minimum description length
  double MDL = std::numeric_limits<double>::max();
  // total number of possible conditions
  std::size_t conditions_count = unique_conditions( X );
  // rule description length
  double RDL = total_description_length( ruleset, X, Y, positive_class,
                                         conditions_count );
  double exceptions = exception_bits( tn, fp, fn, tp );
  RDL -= exceptions;

  while( ! pos_copy.empty() ){

    #ifdef __verbose__
      __logger.log( "Pos: " + std::to_string( pos_copy.size() ) + ", Neg: " +
                    std::to_string( neg_copy.size() ) + " remaining." );
    #endif

    data_split( pos_copy, pos_grow, pos_prune );
    data_split( neg_copy, neg_grow, neg_prune );

    // grow a rule
    #ifdef __verbose__
      __logger.log( "-- Growing" );
    #endif
    auto rule = grow_rule( X, feature_names, pos_grow, neg_grow );
    // prune the rule
    if( m_prune_rules ){
      #ifdef __verbose__
        __logger.log( "-- Pruning rule with size: " + std::to_string( rule.size() ) );
      #endif
      rule = prune_rule( rule, X, pos_prune, neg_prune );
      #ifdef __verbose__
        __logger.log("-- Pruned rule has size: " + std::to_string( rule.size() ) );
      #endif
    }

    RDL += rule_bits( rule, conditions_count );

    std::size_t tp_diff = pos_copy.size();
    std::size_t fp_diff = neg_copy.size();
    pos_copy = rule.not_covered_indices( X, pos_copy );
    neg_copy = rule.not_covered_indices( X, neg_copy );
    
    tp_diff -= pos_copy.size();
    fp_diff -= neg_copy.size();
    tp += tp_diff;
    fp += fp_diff;
    fn -= tp_diff; // = pos_copy.size() should be equal
    tn -= fp_diff; // = neg_copy.size() should be equal

    double exceptions = exception_bits( tn, fp, fn, tp );
    double description_length = RDL + exceptions;

    #ifdef __verbose__
      __logger.log("-- DL: " + std::to_string( description_length ) +
                   ", MDL: " + std::to_string( MDL ) );
    #endif

    if( description_length - MDL > m_difference ){
      #ifdef __verbose__
        __logger.log("-- Description length > MDL, end loop.");
      #endif
      break;
    }
    else if( description_length < MDL )
      MDL = description_length;

    // add to ruleset
    ruleset.add_rule( rule );

  }

  return ruleset;
}

CRuleset CRIPPER::fit( const std::vector<std::vector<double>> & X,
                       const std::vector<std::size_t> & Y,
                       const std::vector<std::string> & feature_names,
                       std::size_t positive_class ){ 

  if( ! X.size() || ! Y.size() )
    throw std::invalid_argument( "Input vectors are empty!" );
  else if( X.front().size() != Y.size() )
    throw std::invalid_argument( "X and Y sizes differ!" );
  else if( X.size() != feature_names.size() )
    throw std::invalid_argument( "Y and feature names differ!" );

  CRuleset ruleset;
  std::vector<std::size_t> pos;
  std::vector<std::size_t> neg;
  pos_neg_split( Y, positive_class, pos, neg );

  ruleset = IREP_star( X, Y, pos, neg, feature_names, positive_class, ruleset );

  for( std::size_t i = 0; i < m_k; ++i ){
    #ifdef __verbose__
      __logger.log( "-- Optimisation #" + std::to_string( i + 1 ) );
    #endif
    // optimise_ruleset
    ruleset = optimise_ruleset( ruleset, X, feature_names, pos, neg );
    auto pos_remaining = ruleset.not_covered_indices( X, pos );
    auto neg_remaining = ruleset.not_covered_indices( X, neg );
    // cover remaining samples
    ruleset = IREP_star( X, Y, pos_remaining, neg_remaining,
                         feature_names, positive_class, ruleset );
    ruleset = generalise_ruleset( ruleset, X, Y, positive_class );
  }

  return ruleset;
}

CRuleset CRIPPER::optimise_ruleset( const CRuleset & input_ruleset,
                                    const std::vector<std::vector<double>> & X,
                                    const std::vector<std::string> & feature_names,
                                    const std::vector<std::size_t> & pos,
                                    const std::vector<std::size_t> & neg ){

  std::vector<std::size_t> pos_copy = pos;
  std::vector<std::size_t> neg_copy = neg;
  std::vector<std::size_t> pos_grow, pos_prune;
  std::vector<std::size_t> neg_grow, neg_prune;
  CRuleset ruleset( input_ruleset );
  std::size_t conditions_count = unique_conditions( X );

  // TODO new part
  std::size_t tn, fp, fn, tp; 
  confusion_matrix( ruleset, 0, X, pos, neg, tn, fp, fn, tp );

  double RDL = 0;
  for( std::size_t i = 0; i < ruleset.size(); ++i )
    RDL += rule_bits( ruleset[i], conditions_count );


  for( std::size_t i = 0; i < input_ruleset.size(); ++i ){

    double exceptions = exception_bits( tn, fp, fn, tp );
    double best_score = std::numeric_limits<double>::max();
    // replacement for tn, fp, fn and tp
    std::size_t tn_r, fp_r, fn_r, tp_r;
    // best score for tn, fp, fn and tp
    std::size_t tn_best, fp_best, fn_best, tp_best;
    tn_best = tn;
    fp_best = fp;
    fn_best = fn;
    tp_best = tp;
    // replacement for RDL and best RDL score
    double RDL_r, RDL_best;
    RDL_best = RDL;    

    CRuleset best_ruleset( ruleset );

    data_split( pos_copy, pos_grow, pos_prune );
    data_split( neg_copy, neg_grow, neg_prune );

    // replacement
    CRule replacement = grow_rule( X, feature_names, pos_grow, neg_grow );
    CRuleset replacement_ruleset( ruleset );
    replacement_ruleset[i] = replacement;
    replacement_ruleset[i] = optimise_prune( replacement_ruleset, i, X,
                                             pos_prune, neg_prune );
    double replacement_TDL = total_description_length(
                               X, replacement_ruleset, ruleset, i,
                               pos_copy, neg_copy,
                               tn, fp, fn, tp,
                               tn_r, fp_r, fn_r, tp_r,
                               RDL, RDL_r,
                               conditions_count
                             );

    if( replacement_TDL < best_score ){
      best_score = replacement_TDL;
      best_ruleset = replacement_ruleset;
      tn_best = tn_r;
      fp_best = fp_r;
      fn_best = fn_r;
      tp_best = tp_r;
      RDL_best = RDL_r;
    }

    // revision
    CRule revision = ruleset[i];
    revision = grow_rule( X, feature_names, pos_grow, neg_grow, revision );
    CRuleset revision_ruleset( ruleset );
    revision_ruleset[i] = revision;
    revision_ruleset[i] = optimise_prune( revision_ruleset, i, X,
                                          pos_prune, neg_prune );
    double revision_TDL = total_description_length(
                            X, revision_ruleset, ruleset, i,
                            pos_copy, neg_copy,
                            tn, fp, fn, tp,
                            tn_r, fp_r, fn_r, tp_r,
                            RDL, RDL_r,
                            conditions_count
                          );

    if( revision_TDL < best_score ){
      best_score = revision_TDL;
      best_ruleset = revision_ruleset;
      tn_best = tn_r;
      fp_best = fp_r;
      fn_best = fn_r;
      tp_best = tp_r;
      RDL_best = RDL_r;
    }

    double original_TDL = exceptions + RDL;

    #ifdef __verbose__
      __logger.log( "-- TDL Scores ... Replacement: " + std::to_string( replacement_TDL ) +
                    ", Revision: " + std::to_string( revision_TDL ) +
                    ", Original: " + std::to_string( original_TDL ) );
    #endif

    if( original_TDL > best_score ){
      #ifdef __verbose__
        __logger.log( "-- Changing rule in ruleset!" );
      #endif
      ruleset = best_ruleset;
      tn = tn_best;
      fp = fp_best;
      fn = fn_best;
      tp = tp_best;
      RDL = RDL_best;
    }

    pos_copy = ruleset[i].not_covered_indices( X, pos_copy );
    neg_copy = ruleset[i].not_covered_indices( X, neg_copy );

  } 

  return ruleset;
}

CRule CRIPPER::optimise_prune( const CRuleset & input_ruleset,
                               std::size_t index,
                               const std::vector<std::vector<double>> & X,
                               const std::vector<std::size_t> & pos_prune,
                               const std::vector<std::size_t> & neg_prune ){
  std::size_t tn, fp, fn, tp;
  confusion_matrix( input_ruleset, index, X, pos_prune, neg_prune,
                    tn, fp, fn, tp );
  double best_val = ( tp + tn ) / ( tp + tn + fp + fn );

  CRule old_rule( input_ruleset[index] );
  CRule rule( old_rule );

  for( auto it = old_rule.o_crbegin(); it != old_rule.o_crend(); ++it ){

    CRuleset ruleset( input_ruleset );
    CRule new_rule( rule );
    new_rule.pop_back();
    ruleset[index] = new_rule;

    confusion_matrix( ruleset, index, X, pos_prune, neg_prune,
                      tn, fp, fn, tp );
    double new_val = ( tp + tn ) / ( tp + tn + fp + fn );
    #ifdef __verbose__
      __logger.log( "---- Old acc: " + std::to_string( best_val ) +
                    ", new acc: " + std::to_string( new_val ) );
    #endif
    if( new_rule.size() && new_val > best_val ){
      best_val = new_val;
      rule = new_rule;
    }
    else
      break;
  }
  return rule;
}

CRuleset CRIPPER::generalise_ruleset( const CRuleset & input_ruleset,
                                      const std::vector<std::vector<double>> & X,
                                      const std::vector<std::size_t> & Y,
                                      std::size_t positive_class ) const{
  std::size_t conditions_count = unique_conditions( X );
  CRuleset best_ruleset( input_ruleset );
  double best_TDL = total_description_length( best_ruleset, X, Y, positive_class,
                                              conditions_count );
  for( std::size_t i = input_ruleset.size() - 1; i < input_ruleset.size(); --i ){
    CRuleset new_ruleset( best_ruleset );
    new_ruleset.pop( i );
    double new_TDL = total_description_length( new_ruleset, X, Y, positive_class,
                                               conditions_count );
    if( new_TDL < best_TDL ){
      #ifdef __verbose__
        __logger.log( "-- Generalise: removed rule with index #" + std::to_string( i ) );
      #endif
      best_ruleset = new_ruleset;
      best_TDL = new_TDL;
    }
  }
  return best_ruleset;
}

CCompetitor::CCompetitor( void ):
    CRuleLearner(){
}

CCompetitor::CCompetitor( double split_ratio, std::size_t random_state,
                          std::size_t categorical_max, std::size_t difference,
                          bool prune_rules, std::size_t n_threads,
                          const std::string & pruning_metric ):
    CRuleLearner( split_ratio, random_state, categorical_max, difference,
                  prune_rules, n_threads, pruning_metric ){
}

CRuleset CCompetitor::fit( const std::vector<std::vector<double>> & X,
                           const std::vector<std::size_t> & Y,
                           const std::vector<std::string> & feature_names,
                           std::size_t positive_class ){ 


  CRuleset ruleset;
  std::vector<std::size_t> pos,neg;
  std::vector<std::size_t> pos_grow, pos_prune;
  std::vector<std::size_t> neg_grow, neg_prune;

  pos_neg_split( Y, positive_class, pos, neg ); 

  // minimum description length
  double MDL = std::numeric_limits<double>::max();
  // total number of possible conditions
  std::size_t conditions_count = unique_conditions( X );
  // rule description length
  double RDL = 0;

  while( ! pos.empty() ){

    #ifdef __verbose__
      __logger.log( "Pos: " + std::to_string( pos.size() ) + ", Neg: " +
                    std::to_string( neg.size() ) + " remaining." );
    #endif

    data_split( pos, pos_grow, pos_prune );
    data_split( neg, neg_grow, neg_prune );

    // grow a rule
    #ifdef __verbose__
      __logger.log( "-- Growing" );
    #endif
    auto rule_grow = grow_rule( X, feature_names, pos_grow, neg_grow );
    auto rule_prune = grow_rule( X, feature_names, pos_prune, neg_prune );
    // prune the rule
    if( m_prune_rules ){
      #ifdef __verbose__
      {
        __logger.log( "-- Pruning rule_grow with size: " + std::to_string( rule_grow.size() ) );
        __logger.log( "-- Pruning rule_prune with size: " + std::to_string( rule_prune.size() ) );

        double metric_val = m_pruning_metric( X, rule_grow, pos_prune, neg_prune );
        __logger.log( "-- rule_grow metric val: " + std::to_string( metric_val ) );
        metric_val = m_pruning_metric( X, rule_prune, pos_grow, neg_grow );
        __logger.log( "-- rule_prune metric val: " + std::to_string( metric_val ) );
      }
      #endif
      rule_grow = prune_rule( rule_grow, X, pos_prune, neg_prune );
      rule_prune = prune_rule( rule_prune, X, pos_grow, neg_grow );
      #ifdef __verbose__
        __logger.log("-- Pruned rule_grow has size: " + std::to_string( rule_grow.size() ) );
        __logger.log("-- Pruned rule_prune has size: " + std::to_string( rule_prune.size() ) );

        double metric_val = m_pruning_metric( X, rule_grow, pos_prune, neg_prune );
        __logger.log( "-- Pruned rule_grow metric val: " + std::to_string( metric_val ) );
        metric_val = m_pruning_metric( X, rule_prune, pos_grow, neg_grow );
        __logger.log( "-- Pruned rule_prune metric val: " + std::to_string( metric_val ) );
      #endif
    }

    CRule rule;
    double grow_val = m_pruning_metric( X, rule_grow, pos_prune, neg_prune );
    double prune_val = m_pruning_metric( X, rule_prune, pos_grow, neg_grow );
    if( grow_val > prune_val )
      rule = rule_grow;
    else
      rule = rule_prune;

    RDL += rule_bits( rule, conditions_count );
    CRuleset review_ruleset( ruleset );
    review_ruleset.add_rule( rule );
    double exceptions = exception_bits( ruleset, X, Y, positive_class );
    double description_length = RDL + exceptions;

    #ifdef __verbose__
      __logger.log("-- DL: " + std::to_string( description_length ) +
                   ", MDL: " + std::to_string( MDL ) );
    #endif

    if( description_length - MDL > m_difference ){
      #ifdef __verbose__
        __logger.log("-- Description length > MDL, end loop.");
      #endif
      break;
    }
    else if( description_length < MDL )
      MDL = description_length;

    pos = rule.not_covered_indices( X, pos );
    neg = rule.not_covered_indices( X, neg );
    // add to ruleset
    ruleset.add_rule( rule );

  }
  return ruleset;
}

COneR::COneR( void ){
}

CRuleset COneR::fit( const std::vector<std::vector<double>> & X,
                     const std::vector<std::size_t> & Y,
                     const std::vector<std::string> & feature_names,
                     std::size_t positive_class ){

  CRuleset best_ruleset;
  double best_acc = std::numeric_limits<double>::lowest();

  for( std::size_t i = 0; i < X.size(); ++i ){

    CRuleset ruleset;
    if( m_categorical_max && unique( X[i] ).size() <= m_categorical_max ){
      /* TODO */
    }
    else{ 
      ruleset = discretise( i, X, Y, feature_names, positive_class );
    }
    std::vector<std::size_t> predictions = predict( ruleset, X );
    double acc = measure_accuracy( Y, predictions );
 
    #ifdef __verbose__
      __logger.log( "Best acc: " + std::to_string( best_acc ) +
                    ", new acc: " + std::to_string( acc ) +
                    ", ruleset size: " + std::to_string( ruleset.size() ) +
                    ", iteration: " + std::to_string( i ) + ", feature: " +
                    feature_names[i] );
    #endif

    if( acc > best_acc ){
      best_acc = acc;
      best_ruleset = ruleset;
    }
  }

  return best_ruleset;
}

std::vector<std::size_t> COneR::predict( const CRuleset & ruleset,
                                         const std::vector<std::vector<double>> & X,
                                         std::size_t positive_class ) const{
  // suppress warning about unused parameter
  (void)positive_class;
  return predict( ruleset, X );
}

std::vector<std::size_t> COneR::predict( const CRuleset & ruleset,
                                         const std::vector<std::vector<double>> & X ) const{

  if( ! X.size() )
    throw std::invalid_argument( "Input vector is empty!" );
  else if( ! ruleset.size() )
    throw std::invalid_argument( "Input ruleset is empty!" );

  std::vector<std::size_t> predictions( X[0].size() );
  std::vector<std::size_t> indices( X[0].size() );
  std::iota( std::begin( indices ), std::end( indices ), 0 ); 

  for( std::size_t i = 0; i < ruleset.size(); ++i ){
    const auto & rule = ruleset[i];
    auto predicted_class = rule.predicted_class();
    auto covered = rule.covered_indices( X, indices );
    indices = rule.not_covered_indices( X, indices );

    for( auto & i : covered )
      predictions[i] = predicted_class;
  }

  return predictions;
}

CRuleset COneR::discretise( std::size_t row,
                            const std::vector<std::vector<double>> & X,
                            const std::vector<std::size_t> & Y,
                            const std::vector<std::string> & feature_names,
                            std::size_t positive_class,
                            std::size_t min_class ) const{

  if( ! X.size() || ! Y.size()  )
    throw std::invalid_argument( "Input vector is empty!" );

  const std::vector<double> & X_row = X[row];

  if( X_row.size() != Y.size() )
    throw std::invalid_argument( "X and Y sizes differ!" ); 

  std::vector<std::size_t> indices = sort_by_indices( X_row ); 

  CRuleset ruleset;
  // a - positive class, b - other class
  std::size_t a = 0, b = 0;
  double curr_val = X_row[indices[0]];
  double last_val = curr_val;

  for( const auto & i : indices ){

    if( curr_val != X_row[i] ){
      if( a >= min_class || b >= min_class ){
        CCondition cond( feature_names[row], row, "range",
                         std::vector<double>{ last_val, curr_val } );
        CRule rule = ( a >= b ? CRule( positive_class, true ) : CRule( positive_class, false ) );

        rule.add_cond( cond );
        ruleset.add_rule( rule );
        last_val = X_row[i];
        a = b = 0;
      }

      curr_val = X_row[i];
    }

    if( Y[i] == positive_class )
      ++a;
    else
      ++b;
  }

  // add last condition
  if( a || b ){
    CCondition cond( feature_names[row], row, "range",
                     std::vector<double>{ last_val, curr_val } );
    CRule rule = ( a >= b ? CRule( positive_class, true ) : CRule( positive_class, false ) );

    rule.add_cond( cond );
    ruleset.add_rule( rule );
  }

  return simplify_ruleset( ruleset, row );
}

CRuleset COneR::simplify_ruleset( const CRuleset & ruleset,
                                  std::size_t row ) const{
  if( ! ruleset.size() )
    return CRuleset();

  CRuleset new_ruleset;
  CRule first = ruleset[0];

  for( std::size_t i = 0; i < ruleset.size(); ++i ){
    if( ( i + 1 < ruleset.size() && ! first.predicts_the_same( ruleset[ i + 1 ] ) ) ||
        ( i == ruleset.size() - 1 && first.predicts_the_same( ruleset[i] ) ) ){
      const CRule & last = ruleset[i];
      auto vec_f = first[row].get_values();
      auto vec_l = last[row].get_values();
    
      std::vector<double> vec_new;
      vec_new.push_back( vec_f[0] );
      vec_new.push_back( vec_l[1] );

      CCondition cond( first[row].get_feature(),
                       first[row].get_index(),
                       "range", vec_new );
      first[row] = cond;
      new_ruleset.add_rule( first );

      if( i + 1 < ruleset.size() )
        first = ruleset[i+1];
    }
  }

  return new_ruleset;
}
#endif /*__rule_learnercpp__*/
