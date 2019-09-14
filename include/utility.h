#ifndef __utility_h__
#define __utility_h__

#include <iostream>      // std :: cout
#include <vector>        // std :: vector
#include <unordered_set> // std :: unordered_set
#include <functional>    // std :: function
#include <regex>         // std :: regex
#include <algorithm>     // std :: copy_n
#include <memory>        // std :: unique_ptr

constexpr int error_file = 1;

void file_error (const std :: string & input);
std :: unique_ptr < int[] > lbl2num (const std :: vector < std :: string > & lbl);


template < class lambda = std :: function < std :: string (std :: string) > >
auto split (const std :: string & txt, const std :: regex & rgx, lambda func = [](std :: string s) -> std :: string { return s; });

#ifdef _MSC_VER
#if (!defined __clang__ && __GNUC__ == 4 && __GNUC_MINOR__ < 9) || __cplusplus < 201400 // no std=c++14 support

namespace std
{
  template < typename type >
  std :: unique_ptr < type > make_unique ( std :: size_t size );
}

#endif
#endif

#endif // __utility_h__
