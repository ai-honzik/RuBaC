#ifdef __verbose__
#ifndef __CLoggercpp__
#define __CLoggercpp__

#include "logger.hpp"

CLogger __logger( true );

CLogger::CLogger( bool to_term ):
    m_to_term( to_term ){

  auto curr_time = std::chrono::system_clock::now();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>( curr_time.time_since_epoch() ).count();

  std::string name = "log_file_" + std::to_string( seconds ) + ".txt";  
  m_file.open( name, std::ios::out );

  
  if( ! m_file.is_open() )
    throw std::runtime_error("Failed to open logfile!");

}

CLogger::~CLogger( void ){
  m_file.close();
}

void CLogger::log( const std::string & message ){
  m_file << message << "\n";
  m_file.flush();
  if( m_to_term )
    std::cout << message << std::endl;
}

#endif /*__CLoggercpp__*/
#endif
