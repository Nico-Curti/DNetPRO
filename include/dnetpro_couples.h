#ifndef __dnetpro_couples_h__
#define __dnetpro_couples_h__

#include <memory>         // std :: unique_ptr

#include <iostream>       // std :: cout
#include <unordered_map>  // std :: unordered_map
#include <vector>         // std :: vector
#include <chrono>         // std :: chrono
#include <climits>        // std :: numeric_limits
#include <numeric>        // std :: accumulate
#include <algorithm>      // std :: sort

#include <score.h>

#ifdef _OPENMP

  #include <omp.h>
  #include <sort.hpp>

#endif

constexpr float epsilon = std :: numeric_limits < float > :: min();  ///< float minimum
constexpr float inf = std :: numeric_limits < float > :: infinity(); ///< float infinity

/**
* @brief Core function for the couple evaluation.
*
* @param data Input matrix of data
* @param Nprobe Number of rows in data (aka number of probes)
* @param Nsample Number of columns in data (aka number of samples)
* @param labels Array of numeric labels
* @param verbose Enable(1)/Disable(0) cout log
* @param percentage Percentage of results to save
* @param nth Number of threads to use in parallel section
*
*
*/
score dnetpro_couples (float ** data,                   // matrix of data
                       const int & Nprobe,              // number of rows in db
                       const int & Nsample,             // number of columns in db
                       int * labels,                    // numeric labels
                       const bool & verbose,            // enable(ON)/disable(OFF) cout log
                       //const bool & return_couples,   // enable(ON)/disable(oFF) return couples(ON)/single(OFF)
                       float percentage = .1f,          // percentage of results to save
                       int nth = -1                     // number of threads to use in parallel section
                       );

#endif // __dnetpro_couples_h__
