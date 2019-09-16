#ifndef __utility_hpp__
#define __utility_hpp__

#include <utility.h>

#ifndef _MSC_VER
#if (!defined __clang__ && __GNUC__ == 4 && __GNUC_MINOR__ < 9) || __cplusplus < 201400 // no std=c++14 support

namespace std
{

template < typename type >
std :: unique_ptr < type > make_unique ( std :: size_t size )
{
  return std :: unique_ptr < type > ( new typename std :: remove_extent < type > :: type[size] () );
}

}

#endif
#endif

#endif // __utility_hpp__
