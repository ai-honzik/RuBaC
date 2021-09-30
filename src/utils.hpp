#ifndef __utilshpp__
#define __utilshpp__

#include <numeric>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "ruleset.hpp"

/** Calculate the base 2 logarithm of n. */
double log_fact( std::size_t n );
/** Calculate the base 2 logarithm of
  * binomial coefficient. */
double log_C( std::size_t n, std::size_t k );
/** Calculate Stirling's approximation of the base 2
  * logarithm of binomial coefficient. */
double Slog_C( std::size_t n, std::size_t k );
/** Calculate the IREP pruning metric. */
double IREP_pruning_metric( const std::vector<std::vector<double>> & X,
                            const CRule & rule,
                            const std::vector<std::size_t> & pos_prune,
                            const std::vector<std::size_t> & neg_prune );
/** Calculate the RIPPER pruning metric. */
double RIPPER_pruning_metric( const std::vector<std::vector<double>> & X,
                              const CRule & rule,
                              const std::vector<std::size_t> & pos_prune,
                              const std::vector<std::size_t> & neg_prune );
/** Calculate the improved RIPPER pruning metric. */
double RIPPER_impr_pruning_metric( const std::vector<std::vector<double>> & X,
                                   const CRule & rule,
                                   const std::vector<std::size_t> & pos_prune,
                                   const std::vector<std::size_t> & neg_prune );
/** Calculate the sqrt RIPPER pruning metric. */
double RIPPER_sqrt_pruning_metric( const std::vector<std::vector<double>> & X,
                                   const CRule & rule,
                                   const std::vector<std::size_t> & pos_prune,
                                   const std::vector<std::size_t> & neg_prune );
/** Calculate the saddle pruning metric. */
double saddle_pruning_metric( const std::vector<std::vector<double>> & X,
                              const CRule & rule,
                              const std::vector<std::size_t> & pos_prune,
                              const std::vector<std::size_t> & neg_prune );

/**
 * @in: vector v
 * @out: sorted indices
 * - sort vector v by indices
 * - sources: https://stackoverflow.com/questions/10580982/c-sort-keeping-track-of-indices
 *            https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
 */
template<typename T>
std::vector<std::size_t> sort_by_indices( const std::vector<T> & v ){

  if( v.size() == 0 )
    return std::vector<std::size_t>();

  std::vector<std::size_t> indices( v.size() );
  std::iota( std::begin( indices ), std::end( indices ), 0 );
  
  std::sort(
      std::begin( indices ), std::end( indices ),
      [&]( std::size_t a, std::size_t b ){ return v[a] < v[b]; }
  );

  return indices;
}

/**
 * @in: vector v, indices of vector v
 * @out: vector with unique values and their counts
 * - calculate the number of occurrences of unique values in v
 * - if idx is not present presume that v is sorted
 * - if idx is present then v is not sorted
 */
template<typename T>
std::vector<std::pair<T,std::size_t>> count_unique_from_sorted(
    const std::vector<T> & v,
    const std::vector<std::size_t> & idx = std::vector<std::size_t>() ){

  if( v.size() == 0 )
    return std::vector<std::pair<T,std::size_t>>();
  else if( idx.size() && v.size() != idx.size() )
    throw std::invalid_argument( "Input vector size and indices differ!" );
  
  std::vector<std::pair<T,std::size_t>> uniq( v.size() );
  
  T curr_val;
  std::size_t cnt = 0;

  if( ! idx.size() ){
    curr_val = v[0];
    for( const auto & x : v ){
      if( x == curr_val )
        ++cnt;
      else{
        uniq.push_back( { curr_val, cnt } );
        curr_val = x;
        cnt = 1;
      }
    }
  }
  else{
    curr_val = v[idx[0]];
    for( const auto & i : idx ){
      if( v[i] == curr_val )
        ++cnt;
      else{
        uniq.push_back( { curr_val, cnt } );
        curr_val = v[i];
        cnt = 1;
      }
    }
  }
  uniq.push_back( { curr_val, cnt } );

  uniq.shrink_to_fit();
  return uniq;
}

/**
  * @in: vector v, indices
  * @out: map with number of occurrences for each T
  * - calculate the number of occurrences using map
  * - if idx is present, use only elements given by it
  */
template<typename T>
std::map<T,std::size_t> unique_counts( const std::vector<T> & v,
                                       const std::vector<std::size_t> & idx =
                                         std::vector<std::size_t>() ){
  if( v.empty() )
    return std::map<T,std::size_t>();

  std::map<T,std::size_t> uniques;

  if( idx.empty() ){
    for( const auto & x : v ){
      auto to_increment = uniques.find( x );
      if( to_increment != uniques.end() )
        to_increment -> second += 1;
      else
        uniques.insert( { x, 1 } );
    }
  }
  else{
    for( const auto & i : idx ){
      auto to_increment = uniques.find( v[i] );
      if( to_increment != uniques.end() )
        to_increment -> second += 1;
      else
        uniques.insert( { v[i], 1 } );
    }
  }

  return uniques;
}

/**
  * @in: vector v, indexes idx
  * @out: set of unique values
  * - using set find unique values in vector v
  * - if idx is present, use only values in v given by idx
  */
template<typename T>
std::set<T> unique( const std::vector<T> & v,
                    const std::vector<std::size_t> & idx =
                      std::vector<std::size_t>() ){

  if( ! v.size() )
    return std::set<T>();

  std::set<T> uniques;

  if( ! idx.size() )
    for( const auto & x : v )
      uniques.insert( x );
  else
    for( const auto & i : idx )
      uniques.insert( v[i] );

  return uniques;

}

/**
  * @in: iterator begin and end
  * @out: modified container
  * - calculate the cumulative sum
  * - modifies the input container (given by the iterators)
  */
template<typename Iterator>
void map_cum_sum_ip( Iterator b,
                     Iterator e ){
  size_t x = 0;
  for( ; b != e; ++b ){
    x += b -> second;
    b -> second = x;
  }
}

#endif /*__utilshpp__*/
