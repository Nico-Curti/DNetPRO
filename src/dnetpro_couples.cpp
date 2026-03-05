#include <dnetpro_couples.h>

std :: unique_ptr < score > dnetpro_couples (
  float ** data,                   // matrix of data
  const int32_t & Nprobe,          // number of rows in db
  const int32_t & Nsample,         // number of columns in db
  int32_t * labels,                // numeric labels
  const bool & verbose,            // enable(ON)/disable(OFF) cout log
  float percentage,                // percentage of results to save
  int32_t nth                      // number of threads to use in parallel section
)
{

  if ( nth == -1 )
  {

#ifdef _OPENMP

    nth = omp_get_max_threads();
    nth -= nth % 2;

#else

    nth = 1;

#endif

  }

  int32_t   Ncomb;                 // total number of combination
  int32_t   Nclass;                // number of classes
  int32_t   topK;                  // number of top-scoring couples to preserve
  int32_t   predict_lbl;           // label predict each time
  int32_t   count;                 // number of variables in LooCV
  int32_t   idx;                   // temporary index of couples
  float  **means    = nullptr;     // matrix loocv of means for each class
  float  **means_sq = nullptr;     // matrix loocv of square means for each class
  float  max_score;                // return value by the classifier
  float  discr;                    // discriminant value
  float  var_a, var_b;             // temporary variable for the variance of classifier
  float  tmp_a, tmp_b;             // temporary variables for classifier
  float  mean_a, mean_b;           // temporary means for classifier
  
  std :: unique_ptr < int32_t[] > idx_sort_couples = nullptr; // sorted indices of couples gene (used for top-M)

  std :: unique_ptr < int32_t[] > row_sum = nullptr; ///< temporary buffer for MCC evaluation
  std :: unique_ptr < int32_t[] > col_sum = nullptr; ///< temporary buffer for MCC evaluation

  std :: unordered_map < int32_t, std :: vector < int32_t > > member_class; // index of member of each class

  for (int32_t i = 0; i < Nsample; ++i)
    member_class[labels[i]].push_back(i);

  Nclass = static_cast < int32_t >(member_class.size());
  Ncomb = (Nprobe * (Nprobe - 1) >> 1);
  if ( percentage <= 0.f ) percentage = .1f;
  topK = std :: max < int32_t >(1, static_cast < int32_t >(Ncomb * percentage));

  if ( verbose )
  {
    std :: cout << "Found " << Nprobe << " probes and " << Nsample << " samples." << std :: endl;
    std :: cout << "Samples per class:" << std :: endl;
    for (const auto & it : member_class) std :: cout << it.first << " : " << it.second.size() << " samples" << std :: endl;
    std :: cout << "Total number of combinations to process " << Ncomb << std :: endl;
    std :: cout << "Number of top-scoring couples to preserve " << topK << std :: endl;
    std :: cout << "Computing Dataset statistics..." << std :: flush;
  }

  idx_sort_couples = std :: make_unique < int32_t[] >(Ncomb);

  // Instance of matrix
  means     = new float * [Nprobe];
  means_sq  = new float * [Nprobe];

  //  score single_gene(Nprobe, Nclass);
  score couples(Ncomb, Nclass);
  auto results = std :: make_unique < score >(topK, Nclass);

  row_sum = std :: make_unique < int32_t[] >(Nclass);
  col_sum = std :: make_unique < int32_t[] >(Nclass);

  auto start_time = std :: chrono :: high_resolution_clock :: now();

  // Compute of means and means_sq of each row for each class
#ifdef _OPENMP

#pragma omp parallel shared (couples, data, means, means_sq) num_threads (nth)
  {
#pragma omp for

#endif
    for (int32_t i = 0; i < Nprobe; ++i)
    {
      means[i]    = new float[Nclass];
      means_sq[i] = new float[Nclass];
      for (const auto & cl : member_class)
      {
        means[i][cl.first]    = std :: accumulate(
          cl.second.begin(), cl.second.end(),
          0.f,
          [& data, & i](const float & val, const int32_t & idx)
          {
            return val + data[i][idx];
          }
        );
        means_sq[i][cl.first] = std :: accumulate(
          cl.second.begin(), cl.second.end(),
          0.f,
          [& data, & i](const float & val, const int32_t & idx)
          {
            return val + data[i][idx] * data[i][idx];
          }
        );
      }
    }

#ifdef _OPENMP
#pragma omp single
    {
#endif

      if (verbose)
      {
        std :: cout << "[done]" << std :: endl;
        std::cout << "Elapsed time "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std::endl;
        std :: cout << "Starting with combinations..." << std :: flush;
      }
      start_time = std :: chrono :: high_resolution_clock :: now();

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
            tmp_a   = (labels[i] == cl.first) ? data[gene_a][i] : 0.f;
            tmp_b   = (labels[i] == cl.first) ? data[gene_b][i] : 0.f;
            count   = (labels[i] == cl.first) ? static_cast < int32_t >(cl.second.size()) - 1 : static_cast < int32_t >(cl.second.size());
            mean_a  = (means[gene_a][cl.first] - tmp_a) / count;
            mean_b  = (means[gene_b][cl.first] - tmp_b) / count;
            var_a   = static_cast < float >(count) / ((means_sq[gene_a][cl.first] - tmp_a * tmp_a) - mean_a * mean_a * count) + epsilon;
            var_b   = static_cast < float >(count) / ((means_sq[gene_b][cl.first] - tmp_b * tmp_b) - mean_b * mean_b * count) + epsilon;
            discr   = -(  (data[gene_a][i] - mean_a) * var_a * (data[gene_a][i] - mean_a)  +
                          (data[gene_b][i] - mean_b) * var_b * (data[gene_b][i] - mean_b)
                       ) // Mahalobis distance
                      // Uncomment for real diagQDA classifier
                      //*.5f
                      //-.5f * (
                      //         std :: log(var_a) + std :: log(var_b)
                      //       )
                      //+ std :: log(static_cast < float >(count) / static_cast < float >(cl.second.size()))
                      ;
            discr       = std :: isnan(discr) ? -inf : discr;
            predict_lbl = (max_score < discr) ? cl.first : predict_lbl;
            max_score   = (max_score < discr) ? discr : max_score;
          }
          predict_lbl = predict_lbl < 0 ? 0 : predict_lbl;

          row_sum[labels[i]] += 1;
          col_sum[predict_lbl] += 1;
          trace += static_cast < int32_t >(labels[i] == predict_lbl);

          couples.class_score[idx * Nclass + labels[i]] += static_cast < int32_t >(labels[i] == predict_lbl); // diagonale per classe
          // confusion matrix flat
          couples.confusion[
            (
              static_cast < std :: size_t >(idx) * static_cast < std :: size_t >(Nclass) + 
              static_cast < std :: size_t >(labels[i])
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
        std :: cout << "[done]" << std :: endl;
        std :: cout << "Elapsed time for sort " << Nprobe << " genes : "
                    << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                    << " sec" << std :: endl;
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
                    << " sec" << std::endl;
        std :: cout << "Starting reordering..." << std :: flush;
        start_time = std :: chrono :: high_resolution_clock :: now();
      }

#ifdef _OPENMP
    }
#endif

  // re-order the couples and the single performances

#ifdef _OPENMP
#pragma omp for
#endif
    for (int32_t i = 0; i < topK; ++i)
    {
      const int32_t idx = idx_sort_couples[i];
      results->tot[i] = couples.tot[idx];
      results->gene_a[i] = couples.gene_a[idx];
      results->gene_b[i] = couples.gene_b[idx];
      results->mcc[i] = couples.mcc[idx];

      const size_t KK = static_cast<size_t>(Nclass) * static_cast<size_t>(Nclass);
      const size_t src_base = static_cast<size_t>(idx) * KK;
      const size_t dst_base = static_cast<size_t>(i)   * KK;

      std :: copy_n(
        couples.confusion.get() + src_base, 
        KK, 
        results->confusion.get() + dst_base
      );

      for (int32_t j = 0; j < Nclass; ++j)
        results->class_score[i * Nclass + j] = couples.class_score[idx * Nclass + j];
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
      }
#ifdef _OPENMP
    }
#endif


#ifdef _OPENMP

  } // end parallel section

#endif

  // delete all useless ptr
  for (int32_t i = 0; i < Nprobe; ++i) 
  {
    delete[] means[i];
    delete[] means_sq[i];
  }

  delete[] means;
  delete[] means_sq;

  return results;

} // end function
