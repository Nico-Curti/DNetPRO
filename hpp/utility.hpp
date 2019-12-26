#ifndef __utility_hpp__
#define __utility_hpp__

#include <utility.h>

#if ( ( __cplusplus < 201100 && !(_MSC_VER) ) || ( __GNUC__ == 4 && __GNUC_MINOR__ < 9) && !(__clang__) )

namespace std
{

  template < typename T >
  std :: unique_ptr < T > make_unique ( std :: size_t size )
  {
    return std :: unique_ptr < T > ( new typename std :: remove_extent < T > :: type[size] () );
  }

}

#endif // __cplusplus

#endif // __utility_hpp__
