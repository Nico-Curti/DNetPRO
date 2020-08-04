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


#if ( ( __cplusplus < 201100 && !(_MSC_VER) ) || ( __GNUC__ == 4 && __GNUC_MINOR__ < 9) && !(__clang__) )

namespace std
{

/**
* @brief wrap for the older version of gcc anc clang
*
* @tparam T type of the pointer array
*
* @param size lenght of unique_ptr array
*
* @returns pointer array as unique_ptr (e.g. std :: unique_ptr < float[] > in modern c++)
*
*/
template < typename T >
std :: unique_ptr < T > make_unique ( std :: size_t size );

}

#endif

#endif // __utility_h__
