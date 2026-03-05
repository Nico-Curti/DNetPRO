/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  The OpenHiP package is licensed under the MIT "Expat" License:
//
//  Copyright (c) 2022: Nico Curti.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  the software is provided "as is", without warranty of any kind, express or
//  implied, including but not limited to the warranties of merchantability,
//  fitness for a particular purpose and noninfringement. in no event shall the
//  authors or copyright holders be liable for any claim, damages or other
//  liability, whether in an action of contract, tort or otherwise, arising from,
//  out of or in connection with the software or the use or other dealings in the
//  software.
//
//M*/

#include <iostream>       // std :: cout
#include <fstream>        // std :: ofstream
#include <iterator>       // std :: istream_iterator
#include <climits>        // std :: numeric_limits
#include <chrono>         // std :: chrono
#include <sstream>        // std :: stringstream
#include <unordered_map>  // std :: unordered_map
#include <numeric>        // std :: accumulate

constexpr float epsilon = std :: numeric_limits < float > :: epsilon ();
constexpr float inf     = std :: numeric_limits < float > :: infinity ();

#include <score.h>        // score object
#include <utils.hpp>      // split function and file_error function
#include <parse_args.hpp> // ArgumentParser object
#include <version.h>

#ifdef _OPENMP

  #include <omp.h>
  #include <sort.hpp>

#endif


void parse_args (const int & argc, char ** argv,
                 bool & omp,
                 std :: string & input_filename,
                 std :: string & output_filename,
                 std :: string & delimiter,
                 float & fraction,
                 bool & bin,
                 bool & verbose,
                 bool & skipID,
                 int32_t & nth
                 )
{
  parser :: ArgumentParser argparse("DNetPRO couples evaluation v" + dnetpro :: info :: get_version());

  argparse.add_flag ("omp", "o", "omp", "Check if the installed DNetPRO supports OpenMP");

  argparse.add_argument < std :: string >("iArg", "f", "input",  "Input filename",              true, "");
  argparse.add_argument < std :: string >("oArg", "o", "output", "Output filename",             true, "");
  argparse.add_argument < std :: string >("dArg", "d", "sep",    "File delimiter",              false, "\t");
  argparse.add_argument < float >(        "fArg", "s", "frac",   "Fraction of results to save", false, 1.f);
  argparse.add_flag ("bArg", "b", "bin",     "Enable Binary output");
  argparse.add_flag ("qArg", "q", "verbose", "Enable stream output");
  argparse.add_flag ("pArg", "p", "probeID", "ProbeID name to skip");

#ifdef _OPENMP
  // round off to the max pow 2
  argparse.add_argument < int32_t >("nArg", "n", "nth", "Number of threads to use", false, omp_get_max_threads() - (omp_get_max_threads() % 2));

#else

  argparse.add_argument < int32_t >("nArg", "n", "nth", "Number of threads to use", false, 1);

#endif

  argparse.parse_args(argc, argv);

  argparse.get("omp", omp);

  argparse.get("iArg", input_filename);
  argparse.get("oArg", output_filename);
  argparse.get("dArg", delimiter);
  argparse.get("fArg", fraction);
  argparse.get("bArg", bin);
  argparse.get("qArg", verbose);
  argparse.get("nArg", nth);
  argparse.get("pArg", skipID);

  return;
}


int main(int argc, char *argv[])
{
  bool  verbose,                                          // enable(ON)/disable(OFF) cout log
        binary,                                           // boolean for the output formatted(OFF)/binary(ON)
        skipID,                                           // skip first element of each row if true (probeID)
        omp = false;                                      // check OMP support
  std :: string delimiter = "\t";                         // file delimiter to use in the split rows
  int32_t Nprobe,                                         // number of rows in db
          Nsample,                                        // number of columns in db
          Ncomb,                                          // total number of combination
          Nclass,                                         // number of classes
          topK,                                           // number of top-scoring couples to preserve
          predict_lbl,                                    // label predict each time
          count,                                          // number of variables in LooCV
          idx,                                            // temporary index of couples
          nth;                                            // number of threads to use in parallel section
  float fraction,                                         // percent of results to save
        **data     = nullptr,                             // matrix of data
        **means    = nullptr,                             // matrix loocv of means for each class
        **means_sq = nullptr,                             // matrix loocv of square means for each class
        max_score,                                        // return value by the classifier
        discr,                                            // discriminant value
        var_a, var_b,                                     // temporary variable for the variance of classifier
        tmp_a, tmp_b,                                     // temporary variables for classifier
        mean_a, mean_b;                                   // temporary means for classifier
  std :: unique_ptr < int32_t[] > num_lbl      = nullptr, // numeric labels
                              idx_sort_single  = nullptr, // sorted indices of single gene
                              idx_sort_couples = nullptr, // sorted indices of couples gene
                              row_sum          = nullptr, // temporary buffer for MCC evaluation
                              col_sum          = nullptr; // temporary buffer for MCC evaluation
  //std :: unique_ptr < float[] > prior = nullptr;
  std :: ifstream is;                                     // input file (db)
  std :: ofstream os;                                     // output of couples

  std :: string input_filename,                           //filename of db
                output_filename,                          //output of couples
                row,                                      // file row
                probeID;                                  // tmp variable for read probeID

  std :: unordered_map < int32_t, std :: vector < int32_t > > member_class; // index of member of each class
  std :: vector < std :: string > labels;                                   // string labels

  // parse the command line input
  parse_args(argc, argv, omp, input_filename, output_filename, delimiter, fraction, binary, verbose, skipID, nth);
  
  if ( omp )
    dnetpro :: info :: get_omp_support ();

  if ( verbose )
    std :: cout << "Using " << nth << " threads" << std :: endl;

  // Open file DB
  is.open(input_filename);

  if(!is) file_error(input_filename);

  // Reading input file
  std :: stringstream buff;
  buff << is.rdbuf();
  is.close();
  buff.unsetf(std :: ios_base :: skipws);
  Nprobe = std::count(  std :: istream_iterator < char >(buff),
                        std :: istream_iterator < char >(),
                        '\n') - 1;  // the number of '\n' is equal to the number of rows
                                    // BUT the first row is of labels

  Ncomb   = (Nprobe * (Nprobe - 1) >> 1);
  
  if ( fraction <= 0.f ) fraction = .1f;
  topK = std :: max < int32_t >(1, static_cast < int32_t >(Ncomb * fraction));

  buff.clear();
  buff.seekg(0, std :: ios :: beg);
  buff.setf(std :: ios_base :: skipws);
  std :: getline(buff, row);                  // read first row of labels
  labels  = split(row, delimiter);
  Nsample = static_cast < int >(labels.size());
  num_lbl = lbl2num(labels);                // convert string to number
  labels.clear();                           // useless variable

  // compute index in label of each class
  for (int32_t i = 0; i < Nsample; ++i) member_class[num_lbl[i]].push_back(i);
  Nclass = static_cast < int32_t >(member_class.size());

  if (verbose)
  {
    std :: cout << "Processing " << input_filename << " dataset."    << std :: endl;
    std :: cout << "Found "      << Nprobe         << " probes and " << Nsample << " samples." << std :: endl;
    std :: cout << "Samples per class:" << std::endl;
    for (const auto & it : member_class) std :: cout << it.first << " : " << it.second.size() << " samples" << std :: endl;
    std :: cout << "Total number of combinations to process " << Ncomb << std :: endl;
    std :: cout << "Reading Dataset..." << std :: flush;
  }
  // setting variables for the next sorting algorithms
#ifdef _OPENMP

  const int32_t diff_size_single  = Nprobe % nth,
                size_single       = diff_size_single  ? Nprobe - diff_size_single  : Nprobe,
                diff_size_couples = Ncomb % nth,
                size_couples      = diff_size_couples ? Ncomb  - diff_size_couples : Ncomb;

#endif

  idx_sort_single  = std :: make_unique < int32_t[] >(Nprobe);
  idx_sort_couples = std :: make_unique < int32_t[] >(Ncomb);

  auto start_time = std :: chrono :: high_resolution_clock :: now();

  // first line (labels) already skipped
  // Instance of matrix
  data      = new float * [Nprobe];
  means     = new float * [Nprobe];
  means_sq  = new float * [Nprobe];
  score single_gene(Nprobe, Nclass);
  score couples(topK, Nclass);

  row_sum = std :: make_unique < int32_t[] >(Nclass);
  col_sum = std :: make_unique < int32_t[] >(Nclass);

  // reading data and computing of means and means_sq
  if (skipID)
    for (int32_t i = 0; i < Nprobe; ++i)
    {
      buff >> probeID;
      data[i] = new float[Nsample];
      std :: transform(data[i], data[i] + Nsample, data[i],
                       [&](float & val)
                       {
                         buff >> val;
                         return val;
                       });
    }
  else
    for (int32_t i = 0; i < Nprobe; ++i)
    {
      data[i] = new float[Nsample];
      std :: transform(data[i], data[i] + Nsample, data[i],
                       [&](float & val)
                       {
                         buff >> val;
                         return val;
                       });
    }
  buff.str(std :: string()); // clear the buffer with an empty string

  // Compute of means and means_sq of each row for each class
#ifdef _OPENMP
#pragma omp parallel shared (single_gene, couples, data, means, means_sq) num_threads (nth)
{
#pragma omp for
#endif
  for (int32_t i = 0; i < Nprobe; ++i)
  {
    means[i]    = new float[Nclass];
    means_sq[i] = new float[Nclass];
    for (const auto & cl : member_class)
    {
      means[i][cl.first]    = std :: accumulate( cl.second.begin(), cl.second.end(),
                                                 0.f,
                                                 [&](const float & val, const int32_t & idx)
                                                 {
                                                   return val + data[i][idx];
                                                 });
      means_sq[i][cl.first] = std :: accumulate( cl.second.begin(), cl.second.end(),
                                                 0.f,
                                                 [&](const float & val, const int32_t & idx)
                                                 {
                                                   return val + data[i][idx] * data[i][idx];
                                                 });
    }
  }

#ifdef _OPENMP
#pragma omp single
  {
#endif

    if (verbose)
    {
      std :: cout << "[done]" << std :: endl;
      std :: cout << "Elapsed time "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std :: endl;
      std :: cout << "Starting with combinations..." << std :: flush;
      start_time = std :: chrono :: high_resolution_clock :: now();
    }

#ifdef _OPENMP
  }
#endif

#ifdef _OPENMP
#pragma omp for private (idx, max_score, predict_lbl, tmp_a, tmp_b, count, mean_a, mean_b, var_a, var_b, discr) collapse (2)
#endif
  for (int32_t gene_a = 0; gene_a < Nprobe; ++gene_a) // for each gene
    for (int32_t gene_b = 0; gene_b < Nprobe; ++gene_b) // for each gene
    {
      if (gene_b <= gene_a) continue;

      idx = ((Nprobe * (Nprobe - 1)) >> 1) - ((Nprobe - gene_a) * ((Nprobe - gene_a) - 1) >> 1) + gene_b - gene_a - 1;
      idx_sort_couples[idx] = idx;

      std :: fill_n(row_sum.get(), Nclass, 0);
      std :: fill_n(col_sum.get(), Nclass, 0);
      int32_t trace = 0;
      
      // Leave One Out Cross Validation with diagQDA
      for (int32_t i = 0; i < Nsample; ++i) // looCV cycle
      {
        max_score   = -inf;
        predict_lbl = -1;
        for (const auto & cl : member_class)
        {
          tmp_a   = (num_lbl[i] == cl.first) ? data[gene_a][i] : 0.f;
          tmp_b   = (num_lbl[i] == cl.first) ? data[gene_b][i] : 0.f;
          count   = (num_lbl[i] == cl.first) ? static_cast < int32_t >(cl.second.size()) - 1 : static_cast < int32_t >(cl.second.size());
          mean_a  = (means[gene_a][cl.first] - tmp_a) / count;
          mean_b  = (means[gene_b][cl.first] - tmp_b) / count;
          var_a   = static_cast < float >(count) / ((means_sq[gene_a][cl.first] - tmp_a * tmp_a) - mean_a * mean_a * count) + epsilon;
          var_b   = static_cast < float >(count) / ((means_sq[gene_b][cl.first] - tmp_b * tmp_b) - mean_b * mean_b * count) + epsilon;
          discr   = - (  (data[gene_a][i] - mean_a) * var_a * (data[gene_a][i] - mean_a)  +
                         (data[gene_b][i] - mean_b) * var_b * (data[gene_b][i] - mean_b)
                      ) // Mahalobis distance
                    // Uncomment for real diagQDA classifier
                    //*.5f
                    //-.5f * (
                    //         std :: log(var_a * var_b)
                    //       )
                    //+ std :: log(static_cast < float >(count) / static_cast < float >(cl.second.size()))
                    ;
          discr   = std :: isnan(discr) ? -inf : discr;
          predict_lbl = (max_score < discr) ? cl.first : predict_lbl;
          max_score   = (max_score < discr) ? discr    : max_score;
        }
        predict_lbl = predict_lbl < 0 ? 0 : predict_lbl;

        row_sum[num_lbl[i]] += 1;
        col_sum[predict_lbl] += 1;
        trace += static_cast < int32_t >(num_lbl[i] == predict_lbl);

        couples.class_score[idx * Nclass + num_lbl[i]] += static_cast < int32_t >(num_lbl[i] == predict_lbl); // diagonale per classe
        // confusion matrix flat
        couples.confusion[
          (
            static_cast < std :: size_t >(idx) * static_cast < std :: size_t >(Nclass) + 
            static_cast < std :: size_t >(num_lbl[i])
          ) * static_cast < std :: size_t >(Nclass)
          + static_cast < std :: size_t >(predict_lbl)
        ] += 1u;
      } // end sample loop

      // update couples information
        couples.gene_a[idx] = gene_a;
        couples.gene_b[idx] = gene_b;
        // total accuracy
        couples.tot[idx] = trace;
        // matthew correlation coefficient
        couples.mcc[idx] = score :: matthews_corrcoef(
          row_sum.get(), col_sum.get(), Nclass, trace, Nsample
        );
    } // end second gene loop

#ifdef _OPENMP
#pragma omp single
  {
#endif

    if (verbose)
    {
      std :: cout << "[done]" << std::endl;
      std :: cout << "Elapsed time for " << Ncomb << " combinations : "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std::endl;
      std :: cout << "Starting with single gene..." << std :: flush;
      start_time = std :: chrono :: high_resolution_clock :: now();
    }

#ifdef _OPENMP
  }
#endif

#ifdef _OPENMP
#pragma omp for private (idx, max_score, predict_lbl, tmp_a, tmp_b, count, mean_a, mean_b, var_a, var_b, discr)
#endif
  for (int32_t gene_a = 0; gene_a < Nprobe; ++gene_a) // for each gene
  {
    idx_sort_single[gene_a] = gene_a;

    std :: fill_n(row_sum.get(), Nclass, 0);
    std :: fill_n(col_sum.get(), Nclass, 0);
    int32_t trace = 0;

    // single gene case
    for (int32_t i = 0; i < Nsample; ++i) // looCV cycle
    {
      max_score   = -inf;
      predict_lbl = -1;
      for (const auto & cl : member_class)
      {
        tmp_a   = (num_lbl[i] == cl.first) ? data[gene_a][i] : 0.f;
        count   = (num_lbl[i] == cl.first) ? static_cast < int32_t >(cl.second.size()) - 1 : static_cast < int32_t >(cl.second.size());
        mean_a  = (means[gene_a][cl.first] - tmp_a) / count;
        var_a   = static_cast < float >(count) / ((means_sq[gene_a][cl.first] - tmp_a * tmp_a) - mean_a * mean_a * count) + epsilon;
        discr   = - (data[gene_a][i] - mean_a) * var_a * (data[gene_a][i] - mean_a) // Mahalobis distance
                  // Uncomment for real diagQDA classifier
                  //- std :: log(var_a)
                  //+ std :: log(static_cast < float >(count) / static_cast < float >(cl.second.size()))
                  ;
        discr       = std :: isnan(discr) ? -inf : discr;
        predict_lbl = (max_score < discr) ? cl.first : predict_lbl;
        max_score   = (max_score < discr) ? discr : max_score;
      }
      predict_lbl = predict_lbl < 0 ? 0 : predict_lbl;

      row_sum[num_lbl[i]] += 1;
      col_sum[predict_lbl] += 1;
      trace += static_cast < int32_t >(num_lbl[i] == predict_lbl);

      single_gene.class_score[gene_a * Nclass + num_lbl[i]] += static_cast < int32_t >(num_lbl[i] == predict_lbl); // diagonale per classe
      // confusion matrix flat
      single_gene.confusion[
        (
          static_cast < std :: size_t >(gene_a) * static_cast < std :: size_t >(Nclass) + 
          static_cast < std :: size_t >(num_lbl[i])
        ) * static_cast < std :: size_t >(Nclass)
        + static_cast < std :: size_t >(predict_lbl)
      ] += 1u;
    } // end sample loop

    // update couples information
    single_gene.gene_a[gene_a] = gene_a;
    single_gene.gene_b[gene_a] = gene_a;
    // total accuracy
    single_gene.tot[gene_a] = trace;
    // matthew correlation coefficient
    single_gene.mcc[gene_a] = score :: matthews_corrcoef(
      row_sum.get(), col_sum.get(), Nclass, trace, Nsample
    );
  } // end first gene loop

  // delete all ptr
#ifdef _OPENMP
#pragma omp single
  {
#endif

    for (int32_t i = 0; i < Nprobe; ++i) delete[] data[i];
    for (int32_t i = 0; i < Nprobe; ++i) delete[] means[i];
    for (int32_t i = 0; i < Nprobe; ++i) delete[] means_sq[i];

    delete[] data;
    delete[] means;
    delete[] means_sq;

#ifdef _OPENMP
  }
#endif

#ifdef _OPENMP
#pragma omp single
    {
#endif
    if (verbose)
    {
      std :: cout << "[done]" << std :: endl;
      std :: cout << "Elapsed time for " << Nprobe << " genes : "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std :: endl;
      std :: cout << "Sorting single gene..." << std :: flush;
      start_time = std :: chrono :: high_resolution_clock :: now();
    }
#ifdef _OPENMP
    }
#endif

#ifdef _OPENMP

#pragma omp single
  {
    mergeargsort_parallel_omp(
      idx_sort_single.get(), single_gene.tot.get(), 0, size_single, nth, 
      [&](const int32_t & a1, const int32_t & a2)
      {
        return single_gene.tot[a1] > single_gene.tot[a2];
      }
    );
    if (diff_size_single)
    {
      std :: sort(
        idx_sort_single.get() + size_single, 
        idx_sort_single.get() + Nprobe, 
        [&](const int32_t & a1, const int32_t & a2)
        {
          return single_gene.tot[a1] > single_gene.tot[a2];
        }
      );
      std :: inplace_merge(
        idx_sort_single.get(), 
        idx_sort_single.get() + size_single, 
        idx_sort_single.get() + Nprobe, 
        [&](const int32_t & a1, const int32_t & a2)
        {
          return single_gene.tot[a1] > single_gene.tot[a2];
        }
      );
    }
  }

#else

  std :: sort(
    idx_sort_single.get(), 
    idx_sort_single.get() + Nprobe, 
    [&](const int32_t & a1, const int32_t & a2)
    {
      return single_gene.tot[a1] > single_gene.tot[a2];
    }
  );

#endif

#ifdef _OPENMP
#pragma omp single
  {
#endif

    if (verbose)
    {
      std :: cout << "[done]" << std::endl;
      std :: cout << "Elapsed time for sort " << Nprobe << " genes : "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std::endl;
      std :: cout << "Sorting couples..." << std :: flush;
      start_time = std :: chrono :: high_resolution_clock :: now();
    }

#ifdef _OPENMP
  }
#endif

#ifdef _OPENMP
#pragma omp single
    {
#endif

    // 1) Partition: keep only TOP M in front (unordered)
    std :: nth_element(
      idx_sort_couples.get(),
      idx_sort_couples.get() + topK,
      idx_sort_couples.get() + Ncomb,
      [&](const int32_t &a1, const int32_t &a2)
      {
        return couples.mcc[a1] > couples.mcc[a2];
      });

#ifdef _OPENMP
    } // single
#endif

// 2) Now sort ONLY the first topK indices (parallel merge-argsort like your logic)
#ifdef _OPENMP
#pragma omp single
    {
      const int32_t diff_size_top = topK % nth;
      const int32_t size_top      = diff_size_top ? (topK - diff_size_top) : topK;

      mergeargsort_parallel_omp(
        idx_sort_couples.get(), couples.mcc.get(), 0, size_top, nth,
        [&](const int32_t &a1, const int32_t &a2)
        {
          return couples.mcc[a1] > couples.mcc[a2];
        });

      if (diff_size_top)
      {
        std :: sort(
          idx_sort_couples.get() + size_top,
          idx_sort_couples.get() + topK,
          [&](const int32_t &a1, const int32_t &a2)
          {
            return couples.mcc[a1] > couples.mcc[a2];
          });

        std :: inplace_merge(
          idx_sort_couples.get(),
          idx_sort_couples.get() + size_top,
          idx_sort_couples.get() + topK,
          [&](const int32_t &a1, const int32_t &a2)
          {
            return couples.mcc[a1] > couples.mcc[a2];
          });
      }
    }
#else
  std :: sort(
    idx_sort_couples.get(),
    idx_sort_couples.get() + topK,
    [&](const int32_t &a1, const int32_t &a2)
    {
      return couples.mcc[a1] > couples.mcc[a2];
    });
#endif

#ifdef _OPENMP
#pragma omp single
  {
#endif

    if (verbose)
    {
      std :: cout << "[done]" << std :: endl;
      std :: cout << "Elapsed time for sort " << Ncomb << " couples : "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std :: endl;
      start_time = std :: chrono :: high_resolution_clock :: now();
      std :: cout << "Writing output results..." << std :: flush;
    }

#ifdef _OPENMP
  }
#endif

#ifdef _OPENMP
} // end parallel section
#endif

  if (binary)
  {
    os.open(output_filename, std :: ios :: out | std :: ios :: binary);
    for (int i = 0; i < Nprobe*fraction; ++i)
    {
      os.write(reinterpret_cast < char* >(&idx_sort_single[i]), sizeof (short int));
      os.write(reinterpret_cast < char* >(&idx_sort_single[i]), sizeof (short int));
      for(int j = 0; j < Nclass; ++j) os.write(reinterpret_cast < char* >(&single_gene.class_score[idx_sort_single[i] * Nclass + j]), sizeof (short int));
      os.write(reinterpret_cast < char* >(&single_gene.tot[idx_sort_single[i]]), sizeof (short int));
      os.write(reinterpret_cast < char* >(&single_gene.mcc[idx_sort_single[i]]), sizeof (float));
    }
    for (int i = 0; i < Ncomb*fraction; ++i)
    {
      os.write(reinterpret_cast < char* >(&idx_sort_couples[i]), sizeof (short int));
      os.write(reinterpret_cast < char* >(&idx_sort_couples[i]), sizeof (short int));
      for (int j = 0; j < Nclass; ++j) os.write(reinterpret_cast < char* >(&couples.class_score[idx_sort_couples[i] * Nclass + j]), sizeof (short int));
      os.write(reinterpret_cast < char* >(&couples.tot[idx_sort_couples[i]]), sizeof (short int));
      os.write(reinterpret_cast < char* >(&couples.mcc[idx_sort_couples[i]]), sizeof (float));
    }
    os.close();
  }
  else
  {
    os.open(output_filename + ".dat");
    for (int i = 0; i < Nprobe*fraction; ++i)
    {
      os  << single_gene.gene_a[idx_sort_single[i]]                                                     << delimiter
          << single_gene.gene_b[idx_sort_single[i]]                                                     << delimiter;
        for (int j = 0; j < Nclass; ++j) os << single_gene.class_score[idx_sort_single[i] * Nclass + j] << delimiter;
      os  << single_gene.tot[idx_sort_single[i]]                                                        << delimiter
          << single_gene.mcc[idx_sort_single[i]]
          << std :: endl;
    }
    for (int i = 0; i < Ncomb*fraction; ++i)
    {
      os  << couples.gene_a[idx_sort_couples[i]]                                                     << delimiter
          << couples.gene_b[idx_sort_couples[i]]                                                     << delimiter;
        for (int j = 0; j < Nclass; ++j) os << couples.class_score[idx_sort_couples[i] * Nclass + j] << delimiter;
      os  << couples.tot[idx_sort_couples[i]]                                                        << delimiter
          << couples.mcc[idx_sort_couples[i]]
          << std :: endl;
    }
    os.close();
  }

  if (verbose)
  {
    std :: cout << "[done]" << std :: endl;
    std :: cout << "Elapsed time for writing "
                << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                << " sec" << std :: endl;
  }

  return 0;
}
