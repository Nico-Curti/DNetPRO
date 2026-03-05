#ifndef __parse_args_hpp__
#define __parse_args_hpp__

#include <parse_args.h>

#ifdef __old_cpp_std__//  __cplusplus < 201700 // no std=c++17 support

  #include <parse_args_old.hpp>

#else // modern version

template < typename >
struct is_std_vector : std :: false_type {};

template <typename T, typename A >
struct is_std_vector < std :: vector < T, A > > : std :: true_type {};

// useful alias
template < typename T >
constexpr bool is_std_vector_v = is_std_vector < T > :: value;


template < typename T > struct inner_type { using type = T; };

template < typename T, typename Alloc >
struct inner_type < std :: vector < T, Alloc > > { using type = typename inner_type < T > :: type; };

// useful alias
template < typename T >
using inner_type_t = typename inner_type < T > :: type;


namespace parser
{

template < typename data_t >
void ArgumentParser :: add_argument (std :: string && name,
  std :: string && short_flag, std :: string && long_flag,
  std :: string && help,
  const bool & req,
  data_t default_value)
{
  static_assert (!std :: is_same_v < data_t, int32_t > ||
                 !std :: is_same_v < data_t, float >   ||
                 !std :: is_same_v < data_t, double >  ||
                 !std :: is_same_v < data_t, char >    ||
                 !std :: is_same_v < data_t, bool >    ||
                 !std :: is_same_v < data_t, std :: string >,
                 "Error parsing! Argument type not understood in command line.");

  if (long_flag == "help")
    this->error_private_flag(long_flag);

  if constexpr ( std :: is_same_v < data_t, std :: string > )
  {
    std :: string string_data_type = "std :: string";
    this->args.emplace_back(argument(static_cast < std :: string && >(name),
      static_cast < std :: string && >(short_flag), static_cast < std :: string && >(long_flag),
      static_cast < std :: string && >(help),
      req,
      false, // is_flag
      static_cast < std :: string && >(default_value),
      static_cast < std :: string && >(string_data_type)));
  }

  else
  {
    std :: string string_data_type = this->type_name < data_t >();
    this->args.emplace_back(argument(static_cast < std :: string && >(name),
      static_cast < std :: string && >(short_flag), static_cast < std :: string && >(long_flag),
      static_cast < std :: string && >(help),
      req,
      false, // is_flag
      static_cast < std :: string && >(std :: to_string(default_value)),
      static_cast < std :: string && >(string_data_type)));
  }

}



template < typename data_t >
void ArgumentParser :: get (const std :: string & name, data_t & values)
{
  static_assert (!std :: is_same_v < data_t, int32_t >                     ||
                 !std :: is_same_v < data_t, float >                       ||
                 !std :: is_same_v < data_t, double >                      ||
                 !std :: is_same_v < data_t, char >                        ||
                 !std :: is_same_v < data_t, bool >                        ||
                 !std :: is_same_v < data_t, std :: string >               ||
                 !std :: is_same_v < data_t, std :: vector < int32_t > >   ||
                 !std :: is_same_v < data_t, std :: vector < float > >     ||
                 !std :: is_same_v < data_t, std :: vector < double > >    ||
                 !std :: is_same_v < data_t, std :: vector < char > >      ||
                 !std :: is_same_v < data_t, std :: vector < bool > >      ||
                 !std :: is_same_v < data_t, std :: vector < std :: string > >,
                 "Error parsing! Variable type unknown by parser."
                );


  if constexpr ( std :: is_same_v < data_t, std :: vector < std :: string > > )
  {
    for (std :: size_t i = 0; i < this->args.size(); ++i)
      if (args[i].name == name)
      {
        values = std :: move(args[i].values);
        return;
      }

    this->error_parsing_unknown_arg(name);
  }

  else if constexpr ( std :: is_same_v < data_t, std :: string > )
  {
    for (std :: size_t i = 0; i < this->args.size(); ++i)
      if (args[i].name == name)
      {
        values = std :: move(args[i].values[0]);
        return;
      }

    this->error_parsing_unknown_arg(name);
  }

  else if constexpr ( std :: is_fundamental_v < data_t > )
  {
    for (std :: size_t i = 0; i < args.size(); ++i)
      if (args[i].name == name)
      {
        try
        {
          values = static_cast < data_t >(std :: stod(args[i].values[0]));
          return;
        }
        catch ( std :: invalid_argument & )
        {
          this->error_parsing_invalid_arg(name, args[i].values[0]);
        }

        catch ( std :: out_of_range & )
        {
          this->error_parsing_out_of_range_arg(name, args[i].values[0]);
        }
      }

    this->error_parsing_unknown_arg(name);
  }

  else if constexpr ( is_std_vector_v < data_t > )
  {
    for (std :: size_t i = 0; i < this->args.size(); ++i)
      if (args[i].name == name)
      {
        // perform variable casting
        for (auto & x : args[i].values)
          values.emplace_back( inner_type_t < data_t >(std :: stod(x) ) );

        return;
      }

    this->error_parsing_unknown_arg(name);
  }

  // string case already skipped
  else
  {
    this->error_parsing_unknown_arg(name);
  }

}



template < typename data_t >
std :: string ArgumentParser :: type_name ()
{
  typedef typename std :: remove_reference < data_t > tr;

  std :: unique_ptr < char, void(*)(void*) > own (


#ifndef _MSC_VER

                                                  abi :: __cxa_demangle (typeid(tr).name(), nullptr, nullptr, nullptr),

#else

                                                  nullptr,

#endif
                                                  std :: free
                                                 );

  std :: string r = own != nullptr ? own.get() : typeid(tr).name();

  if ( std :: is_const < tr > :: value ) r += " const";
  if ( std :: is_volatile < tr > :: value ) r += " volatile";

  if      ( std :: is_lvalue_reference < data_t > :: value ) r += "&";
  else if ( std :: is_rvalue_reference < data_t > :: value ) r += "&&";

  // pretty layout on help message
  std :: regex value_regex("(.*)(std::remove_reference\\<)(.*)(\\>)(.*)");
  r = std::regex_replace(r, value_regex, "$3");

  return r;
}


} // end namespace parser

#endif

#endif // __parse_args_hpp__
