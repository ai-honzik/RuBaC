#ifdef __verbose__
#ifndef __CLoggerhpp__
#define __CLoggerhpp__

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

class CLogger{

  public:
    CLogger( bool to_term=false );
    ~CLogger( void );
    void log( const std::string & message );

  private:
    std::fstream m_file;
    bool m_to_term;
    
};

#endif /*__CLoggerhpp__*/
#endif
