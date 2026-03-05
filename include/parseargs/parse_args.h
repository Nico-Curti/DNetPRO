#ifndef __parse_args_h__
#define __parse_args_h__

#include <memory>   // std :: unique_ptr
#include <iostream> // std :: cerr
#include <utility>  // std :: move
#include <vector>   // std :: vector
#include <string>   // std :: stod
#include <iomanip>  // std :: setw
#include <type_traits> // std :: is_same_v
#include <regex>    // std :: regex
#include <stdexcept>

#ifndef _MSC_VER

  #include <cxxabi.h>

#endif


namespace parser
{

/**
* @class ArgumentParser
* @brief Abstract class for the command line parsing
*
* @details This class implements a command line parser with
* a user interface very closed to the argparse Python package.
* Each argument is manually added with its parameters and flags.
* The magic happens into the parse_args function! After that
* the read arguments can be extracted using the appropriated
* get function.
*/
class ArgumentParser
{
   /**
  * @class argument
  * @brief Abstract class for the command line argument management
  *
  * @details Each command line argument is described as the argument
  * with its parameters
  */
  class argument
  {
    // Private members

  public:

    // Public members

    std :: vector < std :: string > values; ///< list of values associated

    std :: string name; ///< parameter name/unique tag
    std :: string short_flag; ///< short flag for its identification as -f
    std :: string long_flag; ///< long flag for its identification as --flag
    std :: string help; ///< help message
    std :: string defualt_value; ///< default value as string
    std :: string data_type; ///< data type name as string for the print

    bool required; ///< True if the argument is required, False otherwise
    bool is_flag; ///< True if the argument is just a flag, False if it requires a value

    // Constructors

    /**
    * @brief Construct the argument obj.
    *
    * @details The constructor simply allocates the member of the object.
    *
    * @param name Parameter name/tag as unique string.
    * @param short_flag Short flag for the command line identification as -v
    * @param long_flag Long flag for the command line identification as --var
    * @param help Help message for the user
    * @param required Switch on/off if the argument is required or not
    * @param is_flag Switch on/off if the argument requires a value or not
    * @param default_value Default value to use if the parameter is not need
    * @param data_type Data type name as string for the help print
    *
    */
    argument (std :: string && name,
      std :: string && short_flag, std :: string && long_flag,
      std :: string && help,
      const bool & required,
      const bool & is_flag,
      std :: string && default_value,
      std :: string && data_type);

    // Destructors

    /**
    * @brief Destructor.
    *
    * @details Completely delete the object and release the memory of the arrays.
    *
    */
    ~argument () = default;

  };

  // Static Variables

  static int32_t PRINT_HELP; //<<< Error code for the catch management
  static int32_t ERROR_PARSER; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_INPUTS; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_REQUIRED; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_UNKNOWN; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_INVARG; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_OUTRANGE; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_BOOL; //<<< Error code for the catch management
  static int32_t ERROR_PARSER_CHAR; //<<< Error code for the catch management
  static int32_t ERROR_PRIVATE_FLAG; //<<< Error code for the catch management

  // Private Members

  std :: vector < argument > args; ///< list of command line arguments

  std :: string description; ///< program description (useful for the help message)
  std :: string program; ///< program name extracted as argv[0]

public:

  // Constructors

  /**
  * @brief Construct the Argument Parser obj.
  *
  * @details The constructor simply set private variables.
  *
  * @param description Program description name.
  *
  */
  ArgumentParser ( std :: string && description );

  // Destructors

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~ArgumentParser () = default;

  // Public members

  /**
  * @brief Parse the command line arguments.
  *
  * @details This is where the magic happens! The method takes
  * the input command line and internally set the variables related
  * to each required argument.
  *
  * @param argc Number of command line variables.
  * @param argv Series of the command line parameters provided.
  *
  */
  void parse_args (const int32_t & argc, char ** argv);

  // Template methods

  /**
  * @brief Add a new command line pattern to search.
  *
  * @details This method is used to push new command line patterns
  * to search during the argument parsing. Each argument is related to
  * a precise pattern as much as its data type.
  *
  * @param name Parameter name/tag as unique string.
  * @param short_flag Short flag for the command line identification as -v
  * @param long_flag Long flag for the command line identification as --var
  * @param help Help message for the user
  * @param req Switch on/off if the argument is required or not
  * @param default_value Default value to use if the parameter is not need
  *
  */
  template < typename data_t >
  void add_argument (std :: string && name,
    std :: string && short_flag, std :: string && long_flag,
    std :: string && help,
    const bool & req,
    data_t default_value=data_t());

  /**
  * @brief Add a new command line possible flag to search.
  *
  * @details This method is used to push new command line flag (on/off)
  * to search during the argument parsing. The flag argument does not
  * require any value but it is used to enable/disable a bool operation.
  *
  * @param name Parameter name/tag as unique string.
  * @param short_flag Short flag for the command line identification as -v
  * @param long_flag Long flag for the command line identification as --var
  * @param help Help message for the user
  *
  */
  void add_flag (std :: string && name,
    std :: string && short_flag, std :: string && long_flag,
    std :: string && help);

  /**
  * @brief Argument getter.
  *
  * @details This function returns the parameter values parsed from the object
  * setting the value in the variable provided. The variable value is extracted
  * using the tag-name of the variable.
  *
  * @param name Name of the required argument.
  * @param values Value of the variable to set.
  *
  * @tparam Data type of the variable.
  *
  */
  template < typename data_t >
  void get (const std :: string & name, data_t & values);

private:

  // Private methods

  /**
  * @brief Error management utility.
  *
  * @details This function just print the usage of the program with
  * the parameter description. At the end of this function the program
  * exit with the error_index code provided.
  *
  * @param error_index Error code identification for the program exit.
  *
  */
  void print_help (const int32_t & error_index);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  *
  * @param data_type Data type name not understood.
  *
  */
  void error_parsing_type (const std :: string & data_type);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if there are not enough parameters in the command line.
  *
  */
  void error_parsing_inputs_arg ();

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if a required argument is not provided.
  *
  * @param name Name of the required argument.
  *
  */
  void error_parsing_required_arg (const std :: string & name);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises when is provided an argument related to an unknown
  * tag.
  *
  * @param name Name of the required argument.
  *
  */
  void error_parsing_unknown_arg (const std :: string & name);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if an invalid argument is provided.
  *
  * @param name Name of the required argument.
  * @param value Value of the argument.
  *
  */
  void error_parsing_invalid_arg (const std :: string & name, const std :: string & value);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if the argument goes out of range.
  *
  * @param name Name of the required argument.
  * @param value Value of the argument.
  *
  */
  void error_parsing_out_of_range_arg (const std :: string & name, const std :: string & value);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if something goes wrong with the bool casting.
  *
  * @param name Name of the required argument.
  * @param value Value of the argument.
  *
  */
  void error_parsing_bool (const std :: string & name, const std :: string & value);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if something goes wrong with the char casting.
  *
  * @param name Name of the required argument.
  * @param value Value of the argument.
  *
  */
  void error_parsing_char (const std :: string & name, const std :: string & value);

  /**
  * @brief Error management utility.
  *
  * @details This function just print the related error message, printing the
  * help function and finally exiting with the appropriated error.
  * This error raises if a private flag is used by user.
  *
  * @param flag Flag set by user.
  *
  */
  void error_private_flag (const std :: string & flag);

  /**
  * @brief Convert in string the variable dtype.
  *
  * @details This is an utility function for the help message management.
  *
  * @tparam data type
  * @return String of the corresponding data type.
  *
  */
  template < typename data_t >
  std :: string type_name ();

};


} // end namespace parser


#endif // __parse_args_h__
