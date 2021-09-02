#ifndef __rulesethpp__
#define __rulesethpp__

#include <string>
#include <list>
#include <vector>
#include <stdexcept>
#include <map>
#include <set>
#include <ostream>
#include <algorithm>
#include <iterator>

#ifdef __verbose__
  #include "logger.hpp"
  extern CLogger __logger;
#endif

/**
 * (C)Condition is a simple representation of conditions.
 * It implements several methods necessary to work with
 * the conditions, e.g. their modification.
 * Condition can have the following form:
 * - name[3] in { 'a', 'b' }
 * - this translates to: arr[3] == 'a' or arr[3] == 'b',
 *   or with map: arr['name'] == 'a' or arr['name'] == 'b'.
 */
class CCondition{

  public:
    /**
     * @in: feature name, index in matrix,
     *      used operator, given value
     * - create a new condition as follows: feature (index) operator value
     * - e.g.: file_size[5] <= 0.7
     */
    CCondition( const std::string & feature, std::size_t index,
                const std::string & op, double val );
    /**
     * @in: feature name, index in matrix,
     *      used operator, given values
     * - create a new condition as in constructor above
     * - this way one can create conditions with "range" or "in"
     *   operators straight ahead
     * - e.g.: file_size[5] range [0.7, 0.9]
     */
    CCondition( const std::string & feature, std::size_t index,
                const std::string & op, const std::vector<double> & vals );
    /**
     * @in: CCondition
     * - deep copy constructor
     */
    CCondition( const CCondition & src );
    /** return feature name */
    std::string get_feature( void ) const;
    /** return index */
    std::size_t get_index( void ) const;
    /** return operator */
    std::string get_operator( void ) const;
    /** return values */
    std::vector<double> get_values( void ) const;
    /**
     * @in: operator, value
     * @out: true if modified, false otherwise
     * - modify condition, e.g. create range from { >=, <= } operators
     * - e.g. modify( { file_size[5] >= 0.7 }, "<=", 0.9 ) results in
     *   file_size[5] range [0.7, 0.9]
     * - e.g. modify( { name[2] in { 'a', 'b' } }, "in", 'c' ) results in
     *   name[2] in { 'a', 'b', 'c' }
     */
    bool modify( const std::string & op, double val );
    /**
     * @in: condition
     * @out: true if modified, false otherwise
     * - modify condition with source condition in a similar manner
     *   as modify( std::string, double )
     */
    bool modify( const CCondition & src );
    /** compare two conditions */
    bool operator==( const CCondition & x ) const;
    /** deep copy */
    CCondition & operator=( const CCondition & src );
    /** convert condition to string */
    std::string to_string( void ) const;
    /**
     * @in: data, data indices
     * @out: indices covered by a given condition
     * - apply a given condition to the data[input_indices]
     */
    std::vector<std::size_t> covered_indices(
        const std::vector<std::vector<double>> & data,
        const std::vector<std::size_t> & input_indices ) const;
    /**
     * @in: data, data indices
     * @out: indices not covered by a given condition
     * - apply a given !condition to the data[input_indices]
     * - eg. the conditions has m_op = '<=', then the negation
     *   of this operator is '>'
     *   if( x <= v ) would be if( !( x <= v ) ), or if( x > v )
     */
    std::vector<std::size_t> not_covered_indices(
        const std::vector<std::vector<double>> & data,
        const std::vector<std::size_t> & input_indices ) const;

    /** uses to_string() */
    friend std::ostream & operator<<( std::ostream & out,
                                      const CCondition & src );

    /*
     * Following operators are defined:
     * { <=, >=, range, in }.
     * { <=, >=, range } are intended to be used for numerical features.
     *   If the operator is { <=, >= }, then the size of m_con_vals needs to be 1.
     *   If the operator is { range }, then the size of m_con_vals needs to be 2,
     *   at [0] is the lowerbound and at [1] is the upperbound.
     * { in } is intended to be used for categorical features.
     *   If the operator is { in }, then the size of m_cat_vals must be > 0.
     */
    const static std::vector<std::string> Operators;

  private:
    std::string m_f;                // feature
    std::size_t m_ind;                 // index
    std::string m_op;               // operator
    std::vector<double> m_con_vals; // continuous values
    std::set<double> m_cat_vals;    // categorical values

    /**
      * @in: operator (op)
      * @out: true if op is valid operator, false otherwise
      **/
    bool check_operator( const std::string & op ) const;
};

/**
 * (C)Rule represents conjunction of conditions
 * and basic operations necessary to work with them.
 * A rule can have the following form:
 * name[3] in { 'a', 'b' } and file_size[5] <= 0.9.
 */
class CRule{

  public:
    /** basic constructor */
    CRule( void ); //init
    /**
     * @in: predicted class, predict
     * - one can set a rule to predict specific class,
     *   this can be useful if we want to switch predicted
     *   classes in the ruleset (e.g. for 1R)
     */
    CRule( std::size_t pr_class, bool predict );
    /** deep copy constructor */
    CRule( const CRule & src );

    /**
      * @in: condition
      * @out: true if successful, false otherwise
      * - adds a condition to m_cond
      */
    bool add_cond( const CCondition & x );
    bool pop( std::list<std::size_t>::reverse_iterator it );
    bool pop( std::list<std::size_t>::iterator it );
    void pop_back( void );
    std::list<std::size_t> learned_order( void ) const;
    /** returns m_class */
    std::size_t predicted_class( void ) const;
    bool predicts_class( std::size_t pr_class ) const;
    bool predicts_the_same( const CRule & x ) const;
    std::string to_string( void ) const;
    std::size_t size( void ) const;
    bool operator==( const CRule & x ) const;
    CRule & operator=( const CRule & x );
    CCondition & operator[]( std::size_t idx );
    const CCondition & operator[]( std::size_t idx ) const;
    std::vector<std::size_t> covered_indices(
        const std::vector<std::vector<double>> & data,
        const std::vector<std::size_t> & input_indices ) const;
    std::vector<std::size_t> not_covered_indices(
        const std::vector<std::vector<double>> & data,
        const std::vector<std::size_t> & input_indices ) const;

    std::map<std::size_t,CCondition>::iterator i_begin( void );
    std::map<std::size_t,CCondition>::reverse_iterator i_rbegin( void );
    std::map<std::size_t,CCondition>::iterator i_end( void );
    std::map<std::size_t,CCondition>::reverse_iterator i_rend( void );

    std::list<std::size_t>::iterator o_begin( void );
    std::list<std::size_t>::const_iterator o_cbegin( void ) const;
    std::list<std::size_t>::reverse_iterator o_rbegin( void );
    std::list<std::size_t>::const_reverse_iterator o_crbegin( void ) const;
    std::list<std::size_t>::iterator o_end( void );
    std::list<std::size_t>::const_iterator o_cend( void ) const;
    std::list<std::size_t>::reverse_iterator o_rend( void );
    std::list<std::size_t>::const_reverse_iterator o_crend( void ) const;

    friend std::ostream & operator<<( std::ostream & out,
                                      const CRule & src );

  private:
    std::map<std::size_t,CCondition> m_cond; // conditions by indices
    std::list<std::size_t> m_learn_order; // indices in the order in which
                                          // they were learned
    std::size_t m_class; // predicted class
    bool m_predict; // indicates whether this rule predicts m_class or not
    bool m_show_class;

};

//TODO comment
class CRuleset{

  public:
    CRuleset( void ){} //init
    CRuleset( const CRuleset & src );

    bool add_rule( const CRule & x );
    void pop( std::size_t idx );
    std::string to_string( void ) const;
    std::size_t size( void ) const;
    CRuleset & operator=( const CRuleset & src );
    CRule & operator[]( std::size_t idx );
    const CRule & operator[]( std::size_t idx ) const;
    std::vector<std::size_t> covered_indices(
        const std::vector<std::vector<double>> & data,
        const std::vector<std::size_t> & input_indices ) const;
    std::vector<std::size_t> not_covered_indices(
        const std::vector<std::vector<double>> & data,
        const std::vector<std::size_t> & input_indices ) const;
    friend std::ostream & operator<<( std::ostream & out,
                                      const CRuleset & src );

    
  private:
    std::vector<CRule> m_rules;

};
#endif /*__rulesethpp__*/
