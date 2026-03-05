#include <version.h> // this is required due to multiple definitions of version.h

namespace dnetpro
{

namespace info
{

std :: string get_version ()
{
  if ( __dnetpro_major_version__ < 0 || __dnetpro_minor_version__ < 0 || __dnetpro_revision_version__ < 0 )
  {
    std :: cout << "Unknown.Unknown.Unknown" << std :: endl;
    std :: cerr << "Cannot deduce a correct version of DNetPRO! " << std :: endl
                << "Probably something goes wrong with the installation. " << std :: endl
                << "Please use the CMake file provided in the DNetPRO folder project to install the library as described in the project Docs. " << std :: endl
                << "Reference: https://github.com/Nico-Curti/DNetPRO"
                << std :: endl;

    return "Unknown.Unknown.Unknown";
  }

  std :: cout << "DNetPRO version: "
              << __dnetpro_major_version__    << "."
              << __dnetpro_minor_version__    << "."
              << __dnetpro_revision_version__ << std :: endl;

  return std :: to_string (__dnetpro_major_version__) + "." + std :: to_string(__dnetpro_minor_version__) + "." + std :: to_string(__dnetpro_revision_version__);
}

bool get_omp_support ()
{

#ifdef _OPENMP

  std :: cout << "OpenMP support: ENABLED" << std :: endl;
  return true;

#else

  std :: cout << "OpenMP support: DISABLED" << std :: endl;
  return false;

#endif

}

} // end namespace info

} // end namespace dnetpro
