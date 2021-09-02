#ifndef __utilscpp__
#define __utilscpp__

#include "utils.hpp"

double log_fact( std::size_t n ){
  // log( n! ) = log( n ) + log( n - 1 ) + ... + log( 1 )
  // calc log( n! )
  
  long double val = 0.;

  for( std::size_t i = n; i; --i )
    val += log2( i );

  return val;
}

double log_C( std::size_t n, std::size_t k ){
  // C( n, k ) = n! / ( k! * ( n - k )! )
  // log( C( n, k ) ) = log( n! ) - log( k! * ( n - k )! )
  //                  = log( n! ) - ( log( k! ) + log( ( n - k )! ) )

  if( k > n )
    return 0.;
  else if( k == n || k == 0 )
    return 1.; 

  return log_fact( n ) - log_fact( k ) - log_fact( n - k ); 

}

double Slog_C( std::size_t n, std::size_t k ){

  // log( n! ) ~ ( n + 1/2 ) * log( n ) - n + 1/2*log( 2*pi )

  if( k > n )
    return 0.;
  else if( k == n || k == 0 )
    return 1.; 

  double val = ( n + (double)1/2 ) * log2( n );
  val -= ( k + (double)1/2 ) * log2( k );
  val -= ( n - k + (double)1/2) * log2( n - k );
  val -= (double)1/2 * log2( 2 * acos( -1 ) );
  
  return val;

} 

double IREP_pruning_metric( const std::vector<std::vector<double>> & X,
                            const CRule & rule,
                            const std::vector<std::size_t> & pos_prune,
                            const std::vector<std::size_t> & neg_prune ){
  std::size_t P,N,p,n;
  P = pos_prune.size();
  N = neg_prune.size();
  p = rule.covered_indices( X, pos_prune ).size();
  n = rule.covered_indices( X, neg_prune ).size();

  if( P < 1 && N < 1 )
    return 0.;

  double ret_val = (double)( p + ( N - n ) )/( P + N );
  return ret_val;

}

double RIPPER_pruning_metric( const std::vector<std::vector<double>> & X,
                              const CRule & rule,
                              const std::vector<std::size_t> & pos_prune,
                              const std::vector<std::size_t> & neg_prune ){
  long long int p,n;
  // TODO safe typecast?
  p = (long long int)rule.covered_indices( X, pos_prune ).size();
  n = (long long int)rule.covered_indices( X, neg_prune ).size();

  if( p < 1 && n < 1 )
    return 0.;

  double ret_val = (double)( p - n ) / ( p + n );
  return ret_val;

}
#endif /*__utilscpp__*/
