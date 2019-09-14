#ifndef __utility_hpp__
#define __utility_hpp__

#include <utility.h>

template < class lambda >
auto split (const std :: string & txt, const std :: regex & rgx, lambda func)
{
  using type = typename std :: result_of < decltype(func)(std :: string) > :: type;

  std :: sregex_token_iterator beg(txt.begin(), txt.end(), rgx, -1);
  std :: sregex_token_iterator end;

  std :: size_t ntoken = std :: distance(beg, end);

  std :: vector<type> token(ntoken);
  std :: generate(token.begin(), token.end(),
                  [&] () mutable
                  {
                    return func(*(beg++));
                  });

  return token;
}

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

#endif // __utility_hpp__
