#ifndef __utility_h__
#define __utility_h__

#include <iostream>      // std :: cout
#include <string>        // std :: string
#include <vector>        // std :: vector
#include <unordered_set> // std :: unordered_set
#include <algorithm>     // std :: copy_n
#include <memory>        // std :: unique_ptr

constexpr int error_file = 1;

void file_error (const std :: string & input);
std :: unique_ptr < int[] > lbl2num (const std :: vector < std :: string > & lbl);


std :: vector < std :: string > split (const std :: string & txt, const std :: string & del);


#ifndef _MSC_VER
#if (!defined __clang__ && __GNUC__ == 4 && __GNUC_MINOR__ < 9) || __cplusplus < 201400 // no std=c++14 support

namespace std
{
  template < typename type >
  std :: unique_ptr < type > make_unique ( std :: size_t size );
}

#endif
#endif

#endif // __utility_h__
