#ifndef __rulesetcpp__
#define __rulesetcpp__

#include "./ruleset.hpp"

const std::vector<std::string> CCondition::Operators{
  "<=", ">=","range", "in"
};

CCondition::CCondition( const std::string & feature, std::size_t index,
                        const std::string & op, double val ):
    m_f( feature ), m_ind( index ){

  //check operator
  if( check_operator( op ) && op != "range" )
    m_op = op;
  else
    throw std::invalid_argument("Wrong operator!");

  if( m_op == "in" )
    m_cat_vals.insert( val );
  else
    m_con_vals.push_back( val );
}

CCondition::CCondition( const std::string & feature, std::size_t index,
                        const std::string & op, const std::vector<double> & vals ):
    m_f( feature ), m_ind( index ){

  if( op != "range" && op != "in" )
    throw std::invalid_argument("Wrong operator!");

  m_op = op;

  if( op == "range" ){

    if( vals.size() != 2 )
      throw std::length_error("Invalid vector length!");

    m_con_vals = vals;
  }
  else if( op == "in" ){
    
    if( vals.size() == 0 )
      throw std::length_error("Invalid vector length!");
     
    for( const auto & x: vals )
      m_cat_vals.insert( x ); 
  }
}

CCondition::CCondition( const CCondition & src ){
  m_f = src.m_f;
  m_ind = src.m_ind;
  m_op = src.m_op;
  m_con_vals = src.m_con_vals;
  m_cat_vals = src.m_cat_vals;
}

std::string CCondition::get_feature( void ) const{
  return m_f;
}

std::size_t CCondition::get_index( void ) const{
  return m_ind;
}

std::string CCondition::get_operator( void ) const{
  return m_op;
}

std::vector<double> CCondition::get_values( void ) const{
  if( m_op == "in" )
    return std::vector<double>( m_cat_vals.begin(), m_cat_vals.end() );

  return m_con_vals;
}

bool CCondition::modify( const std::string & op, double val ){

  // check the input operator
  if( ! check_operator( op ) || op == "range" ){
    throw std::invalid_argument("Wrong operator!");
    return false;
  }

  if( m_op == "range" ){
    // change lowerbound
    if( op == ">=" && m_con_vals.size() == 2 && val > m_con_vals[0] )
      m_con_vals[0] = val;
    // change upperbound
    else if( op == "<=" && m_con_vals.size() == 2 && val < m_con_vals[1] )
      m_con_vals[1] = val;
    // other operators than { <=, >= } cannot be combined with
    // { range }
    else{
      throw std::invalid_argument("Bad operator combination!");
      return false;
    }
  }
  else if( op == ">=" ){
    // change lowerbound
    if( m_op == ">=" && m_con_vals.size() == 1 && val > m_con_vals[0] )
      m_con_vals[0] = val;
    // add upperbound, one may create an invalid condition,
    // such as x >= 3 && x <= 2;
    // change the inner operator type
    else if( m_op == "<=" && m_con_vals.size() == 1 ){
      m_op = "range";
      m_con_vals.insert( m_con_vals.begin(), val );
    }
  }
  else if( op == "<=" ){
    // change upperbound
    if( m_op == "<=" && m_con_vals.size() == 1 && val < m_con_vals[0] )
      m_con_vals[0] = val;
    // add lowerbound, see above comment for `op == ">="`
    else if( m_op == ">=" && m_con_vals.size() == 1 ){
      m_op = "range";
      m_con_vals.push_back( val );
    }
  }
  else if( op == "in" && m_op == "in" )
    return m_cat_vals.insert( val ).second;
  else{
    throw std::invalid_argument("Could not deduce operator use!");
    return false;
  }

  return true;

}

bool CCondition::modify( const CCondition & src ){

  bool a = true;

  if( src.m_op == "range" ){
    //TODO may leave the condition inconsistent?
    a &= modify( ">=", src.m_con_vals[0] );
    a &= modify( "<=", src.m_con_vals[1] );
  }
  else if( src.m_op == "in" )
    m_cat_vals.insert( src.m_cat_vals.begin(),
                       src.m_cat_vals.end() );
  else if( src.m_op == "<=" || src.m_op == ">=" )
    a &= modify( src.m_op, src.m_con_vals.front() );
  else
    return false;
  
  return a;

}

bool CCondition::operator==( const CCondition & x ) const{

  if( &x == this )
    return true;

  if( m_f == x.m_f && m_ind == x.m_ind && m_op == x.m_op &&
      m_con_vals == x.m_con_vals &&
      m_cat_vals == x.m_cat_vals )
    return true;

  return false;
}

CCondition & CCondition::operator=( const CCondition & src ){
  if( &src == this )
    return *this;

  m_con_vals.clear();
  m_cat_vals.clear();

  m_f = src.m_f;
  m_ind = src.m_ind;
  m_op = src.m_op;
  m_con_vals = src.m_con_vals;
  m_cat_vals = src.m_cat_vals;

  return *this;
}

std::string CCondition::to_string( void ) const{

  std::string out;
  out = m_f + "[" + std::to_string( m_ind ) + "] " + m_op;
  out += " ";

  if( m_op == ">=" || m_op == "<=" )
    out += std::to_string( m_con_vals.front() );
  else if( m_op == "range" )
    out += "[" + std::to_string( m_con_vals[0] ) + ", " +
           std::to_string( m_con_vals[1] ) + "]";
  else if( m_op == "in" ){
    out += "{";
    for( const auto & x: m_cat_vals )
      out += " " + std::to_string( x ) + ",";
    out.pop_back();
    out += " }";
  }

  return out;
}

std::vector<std::size_t> CCondition::covered_indices(
    const std::vector<std::vector<double>> & data,
    const std::vector<std::size_t> & input_indices ) const{

  // TODO prefixed size? e.g. 1/2 of input_indices.size()
  std::vector<std::size_t> indices;
  auto & row = data[m_ind];

  // determine which condition ( <= ... ) needs to be used
  //   for x in indices
  //     for every index that the condition applies to insert it
  //     into a vector
  // even though this looks kinda ugly we don't want to compare
  // the operator each time
  if( m_op == "<=" ){
    auto & upper = m_con_vals.front();
    for( const auto & i : input_indices )
      if( row[i] <= upper )
        indices.push_back( i );
  }
  else if( m_op == ">=" ){
    auto & lower = m_con_vals.front();
    for( const auto & i : input_indices )
      if( row[i] >= lower )
        indices.push_back( i );
  }
  else if( m_op == "range" ){
    auto & lower = m_con_vals[0];
    auto & upper = m_con_vals[1];
    for( const auto & i : input_indices )
      if( row[i] >= lower && row[i] <= upper )
        indices.push_back( i );
  }
  else if( m_op == "in" ){
    for( const auto & i : input_indices )
      for( const auto & val : m_cat_vals )
        if( row[i] == val ){
          indices.push_back( i );
          break;
        }
  }
  else
    throw std::runtime_error( "Unknown operator encountered" );

  return indices;
}

std::vector<std::size_t> CCondition::not_covered_indices(
    const std::vector<std::vector<double>> & data,
    const std::vector<std::size_t> & input_indices ) const{

  // TODO prefixed size? e.g. 1/2 of input_indices.size()
  std::vector<std::size_t> indices;
  auto & row = data[m_ind];

  if( m_op == "<=" ){
    auto & upper = m_con_vals.front();
    for( const auto & i : input_indices )
      if( !( row[i] <= upper ) )
        indices.push_back( i );
  }
  else if( m_op == ">=" ){
    auto & lower = m_con_vals.front();
    for( const auto & i : input_indices )
      if( !( row[i] >= lower ) )
        indices.push_back( i );
  }
  else if( m_op == "range" ){
    auto & lower = m_con_vals[0];
    auto & upper = m_con_vals[1];
    for( const auto & i : input_indices )
      if( !( row[i] >= lower && row[i] <= upper ) )
        indices.push_back( i );
  }
  else if( m_op == "in" ){
    for( const auto & i : input_indices ){
      bool flag = false;
      for( const auto & val : m_cat_vals )
        if( row[i] == val ){
          flag = true;
          break;
        }
      if( ! flag )
        indices.push_back( i );
    }
  }
  else
    throw std::runtime_error( "Unknown operator encountered" );

  return indices;
}
std::ostream & operator<<( std::ostream & out, const CCondition & src ){

  out << src.to_string();

  return out;
}

bool CCondition::check_operator( const std::string & op ) const{

  //check available operators
  for( const auto & x: CCondition::Operators )
    if( x == op )
      return true;
  
  return false;
}

CRule::CRule( void ):
    m_show_class( false ){
}

CRule::CRule( std::size_t pr_class, bool predict ):
    m_class( pr_class ), m_predict( predict ), m_show_class( true ){
} 

CRule::CRule( const CRule & src ):
    m_cond( src.m_cond ), m_learn_order( src.m_learn_order ),
    m_class( src.m_class ), m_predict( src.m_predict ),
    m_show_class( src.m_show_class ){
}

bool CRule::add_cond( const CCondition & x ){

  auto it = m_cond.find( x.get_index() );

  if( it != m_cond.end() ){
    it -> second . modify( x );
    return false;
  }
  else{
    m_cond.insert( it, { x.get_index(), x } );
    m_learn_order.push_back( x.get_index() );
  }

  return true;
}

bool CRule::pop( std::list<std::size_t>::reverse_iterator it ){

  if( it == m_learn_order.rend() ){
    return false;
  }

  std::size_t val = *it;
  m_learn_order.erase( std::next( it ).base() );
  m_cond.erase( val );

  return true;
}

bool CRule::pop( std::list<std::size_t>::iterator it ){

  if( it == m_learn_order.end() ){
    return false;
  }

  std::size_t val = *it;
  m_learn_order.erase( it );
  m_cond.erase( val );

  return true;
}

void CRule::pop_back( void ){
  auto to_erase = m_learn_order.back();
  m_learn_order.pop_back();
  m_cond.erase( to_erase );
}

std::list<std::size_t> CRule::learned_order( void ) const{
  return m_learn_order;
}

std::size_t CRule::predicted_class( void ) const{

  if( ! m_show_class )
    throw std::runtime_error( "Prediction undefined!" );

  if( m_predict )
    return m_class;

  return !m_class;

}

bool CRule::predicts_class( std::size_t pr_class ) const{
  return !( ( pr_class == m_class ) ^ m_predict );
}

bool CRule::predicts_the_same( const CRule & x ) const{

  if( ! m_show_class || ! x.m_show_class )
    throw std::runtime_error( "Predictions undefined!" );

  if( m_class == x.m_class && m_predict == x.m_predict )
    return true;

  return false;

}

std::string CRule::to_string( void ) const{

  std::string out;

  for( const auto & x: m_cond )
    out += x.second.to_string() + " && ";
  out = out.substr( 0, out.size() - 4 );
  
  if( m_show_class ){
    out += " -> is ";
    if( ! m_predict )
      out += "not ";
    out += std::to_string( m_class );
  }
  
  return out;
}

std::size_t CRule::size( void ) const{
  return m_cond.size();
}

bool CRule::operator==( const CRule & x ) const{

  if( m_show_class == x.m_show_class && m_cond == x.m_cond ){
    if( m_show_class &&
        ( m_class != x.m_class || m_predict != x.m_predict ) )
      return false;

    return true;
  }

  return false;

}

CRule & CRule::operator=( const CRule & x ){

  if( &x == this )
    return *this;

  m_cond = x.m_cond;
  m_learn_order = x.m_learn_order;
  m_class = x.m_class;
  m_predict = x.m_predict;
  m_show_class = x.m_show_class;

  return *this;
}

CCondition & CRule::operator[]( std::size_t idx ){
  return m_cond.at( idx );
}

const CCondition & CRule::operator[]( std::size_t idx ) const{
  return m_cond.at( idx );
}

std::vector<std::size_t> CRule::covered_indices( 
    const std::vector<std::vector<double>> & data,
    const std::vector<std::size_t> & input_indices ) const{

  std::vector<std::size_t> indices = input_indices;

  for( const auto & c : m_cond )
    indices = c.second.covered_indices( data, indices ); 
  
  return indices;
}

std::vector<std::size_t> CRule::not_covered_indices(
    const std::vector<std::vector<double>> & data,
    const std::vector<std::size_t> & input_indices ) const{

  std::vector<std::size_t> indices =
    covered_indices( data, input_indices );
  std::vector<std::size_t> diff;

  // TODO requires the indices to be sorted!
  // so far the operations should keep them in order
  std::set_difference( input_indices.begin(), input_indices.end(),
                       indices.begin(), indices.end(),
                       std::inserter( diff, diff.begin() ) );

  return diff;
}

std::map<std::size_t,CCondition>::iterator CRule::i_begin( void ){
  return m_cond.begin();
}
std::map<std::size_t,CCondition>::reverse_iterator CRule::i_rbegin( void ){
  return m_cond.rbegin();
}
std::map<std::size_t,CCondition>::iterator CRule::i_end( void ){
  return m_cond.end();
}
std::map<std::size_t,CCondition>::reverse_iterator CRule::i_rend( void ){
  return m_cond.rend();
}

std::list<std::size_t>::iterator CRule::o_begin( void ){
  return m_learn_order.begin(); 
}
std::list<std::size_t>::const_iterator CRule::o_cbegin( void ) const{
  return m_learn_order.cbegin(); 
}
std::list<std::size_t>::reverse_iterator CRule::o_rbegin( void ){
  return m_learn_order.rbegin();
}
std::list<std::size_t>::const_reverse_iterator CRule::o_crbegin( void ) const{
  return m_learn_order.crbegin();
}
std::list<std::size_t>::iterator CRule::o_end( void ){
  return m_learn_order.end();
}
std::list<std::size_t>::const_iterator CRule::o_cend( void ) const{
  return m_learn_order.cend();
}
std::list<std::size_t>::reverse_iterator CRule::o_rend( void ){
  return m_learn_order.rend();
}
std::list<std::size_t>::const_reverse_iterator CRule::o_crend( void ) const{
  return m_learn_order.crend();
}

std::ostream & operator<<( std::ostream & out, const CRule & src ){
  out << src.to_string();
  return out;
}

CRuleset::CRuleset( const CRuleset & src ):
    m_rules( src.m_rules ){
}

bool CRuleset::add_rule( const CRule & x ){

  //check if last two rules are the same
  if( m_rules.size() > 0 && m_rules.back() == x )
    return false;

  m_rules.push_back( x );

  return true;
}

void CRuleset::pop( std::size_t idx ){

  if( idx >= m_rules.size() )
    throw std::invalid_argument( "Index out of range" );

  m_rules.erase( std::next( m_rules.begin(), idx ) );
}

std::string CRuleset::to_string( void ) const{

  if( m_rules.empty() )
    return "[ empty ]";

  std::string out;
  out = "[\n";
  for( const auto & x: m_rules )
    out += x.to_string() + ",\n";
  out = out.substr( 0, out.size() - 2 ) + "\n]";

  return out;
}

std::size_t CRuleset::size( void ) const{
  return m_rules.size();
}

CRuleset & CRuleset::operator=( const CRuleset & src ){

  if( &src == this )
    return *this;

  m_rules = src.m_rules;
  return *this;
}

CRule & CRuleset::operator[]( std::size_t idx ){
  return m_rules[idx];
}

const CRule & CRuleset::operator[]( std::size_t idx ) const{
  return m_rules[idx];
}

std::vector<std::size_t> CRuleset::covered_indices(
    const std::vector<std::vector<double>> & data,
    const std::vector<std::size_t> & input_indices ) const{

  std::vector<std::size_t> indices =
    not_covered_indices( data, input_indices );

  std::vector<std::size_t> diff;

  // TODO requires the indices to be sorted!
  std::set_difference( input_indices.begin(), input_indices.end(),
                       indices.begin(), indices.end(),
                       std::inserter( diff, diff.begin() ) );

  return diff;  
}

std::vector<std::size_t> CRuleset::not_covered_indices(
    const std::vector<std::vector<double>> & data,
    const std::vector<std::size_t> & input_indices ) const{

  // indices need to be modified throughout the process
  std::vector<std::size_t> indices = input_indices;

  // TODO parallelizable?
  for( const auto & r: m_rules )
    indices = r.not_covered_indices( data, indices );

  return indices;
}
std::ostream & operator<<( std::ostream & out, const CRuleset & src ){
  out << src.to_string();
  return out;
}

#endif /*__rulesetcpp__*/
