#include <dnetpro_couples.h>

score dnetpro_couples (float ** data,                   // matrix of data
                       const int & Nprobe,              // number of rows in db
                       const int & Nsample,             // number of columns in db
                       int * labels,                    // numeric labels
                       const bool & verbose,            // enable(ON)/disable(OFF) cout log
                       //const bool & return_couples,   // enable(ON)/disable(oFF) return couples(ON)/single(OFF)
                       float percentage,                // percentage of results to save
                       int nth                          // number of threads to use in parallel section
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


  int   Ncomb;                    // total number of combination
  int   Nclass;                   // number of classes
  int   predict_lbl;              // label predict each time
  int   count;                    // number of variables in LooCV
  int   idx;                      // temporary index of couples
  float **means    = nullptr;     // matrix loocv of means for each class
  float  **means_sq = nullptr;     // matrix loocv of square means for each class
  float  max_score;                // return value by the classifier
  float  discr;                    // discriminant value
  float  var_a, var_b;             // temporary variable for the variance of classifier
  float  tmp_a, tmp_b;             // temporary variables for classifier
  float  mean_a, mean_b;           // temporary means for classifier
  //std :: unique_ptr < int[] > idx_sort_single  = nullptr; // sorted indices of single gene
  std :: unique_ptr < int[] > idx_sort_couples = nullptr; // sorted indices of couples gene

  std :: unordered_map < int, std :: vector < int > > member_class; // index of member of each class

  for (int i = 0; i < Nsample; ++i)
    member_class[labels[i]].push_back(i);

  Nclass = static_cast < int >(member_class.size());
  Ncomb = (Nprobe * (Nprobe - 1) >> 1);

  if ( verbose )
  {
    std :: cout << "Found " << Nprobe << " probes and " << Nsample << " samples." << std :: endl;
    std :: cout << "Samples per class:" << std :: endl;
    for (const auto & it : member_class) std :: cout << it.first << " : " << it.second.size() << " samples" << std :: endl;
    std :: cout << "Total number of combination to process " << Ncomb << std :: endl;
    std :: cout << "Computing Dataset statistics..." << std :: flush;
  }

  // setting variables for the next sorting algorithms
#ifdef _OPENMP

  //const int diff_size_single  = Nprobe % nth;
  //const int size_single       = diff_size_single  ? Nprobe - diff_size_single  : Nprobe;
  const int diff_size_couples = Ncomb % nth;
  const int size_couples      = diff_size_couples ? Ncomb  - diff_size_couples : Ncomb;

#endif

//  idx_sort_single  = std :: make_unique < int[] >(Nprobe);
  idx_sort_couples = std :: make_unique < int[] >(Ncomb);

  // Instance of matrix
  means     = new float * [Nprobe];
  means_sq  = new float * [Nprobe];

//  score single_gene(Nprobe, Nclass);
  score couples(Ncomb, Nclass);
  score results(static_cast < int >(Ncomb  * percentage), Nclass);

  auto start_time = std :: chrono :: high_resolution_clock :: now();

  // Compute of means and means_sq of each row for each class
#ifdef _OPENMP

#pragma omp parallel shared (/*single_gene,*/ couples, data, means, means_sq) num_threads (nth)
  {
#pragma omp for

#endif
    for (int i = 0; i < Nprobe; ++i)
    {
      means[i]    = new float[Nclass];
      means_sq[i] = new float[Nclass];
      for (const auto & cl : member_class)
      {
        means[i][cl.first]    = std :: accumulate( cl.second.begin(), cl.second.end(),
                                                   0.f,
                                                   [& data, & i](const float & val, const int & idx)
                                                   {
                                                     return val + data[i][idx];
                                                   });
        means_sq[i][cl.first] = std :: accumulate( cl.second.begin(), cl.second.end(),
                                                   0.f,
                                                   [& data, & i](const float & val, const int & idx)
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
        std::cout << "Elapsed time "
                  << std :: chrono :: duration_cast < std :: chrono :: seconds >(std :: chrono :: high_resolution_clock :: now() - start_time).count()
                  << " sec" << std::endl;
        std :: cout << "Starting with the combination..." << std :: flush;
      }
      start_time = std :: chrono :: high_resolution_clock :: now();

#ifdef _OPENMP
    }
#endif

#ifdef _OPENMP
#pragma omp for private (idx, max_score, predict_lbl, tmp_a, tmp_b, count, mean_a, mean_b, var_a, var_b, discr) collapse (2)
#endif
    for (int gene_a = 0; gene_a < Nprobe; ++gene_a) // for each gene
      for (int gene_b = 0; gene_b < Nprobe; ++gene_b) // for each gene
      {
        if (gene_b <= gene_a) continue;

        idx = ((Nprobe * (Nprobe - 1)) >> 1) - ((Nprobe - gene_a) * ((Nprobe - gene_a) - 1) >> 1) + gene_b - gene_a - 1;
        idx_sort_couples[idx] = idx;
        // Leave One Out Cross Validation with diagQDA
        for (int i = 0; i < Nsample; ++i) // looCV cycle
        {
          max_score   = -inf;
          predict_lbl = -1;
          for (const auto & cl : member_class)
          {
            tmp_a   = (labels[i] == cl.first) ? data[gene_a][i] : 0.f;
            tmp_b   = (labels[i] == cl.first) ? data[gene_b][i] : 0.f;
            count   = (labels[i] == cl.first) ? static_cast < int >(cl.second.size()) - 1 : static_cast < int >(cl.second.size());
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
            predict_lbl = (max_score < discr) ? cl.first : predict_lbl;
            max_score   = (max_score < discr) ? discr : max_score;
          }
          couples.class_score[predict_lbl][idx] += static_cast < int >(labels[i] == predict_lbl);
        } // end sample loop

        // update total and gene number
        couples.gene_a[idx] = gene_a;
        couples.gene_b[idx] = gene_b;
        couples.tot[idx]  = std :: accumulate(  couples.class_score.get(), couples.class_score.get() + couples.n_class,
                                                 0, [&idx](const int & res, std :: unique_ptr < int[] > & score)
                                                 {
                                                   return res + score[idx];
                                                 });
        couples.mcc[idx]  = score :: matthews_corrcoef(couples.class_score[0][idx], static_cast < int >(member_class[0].size()), couples.class_score[1][idx], static_cast < int >(member_class[1].size()));
      } // end second gene loop
/*
#ifdef _OPENMP
#pragma omp for private (idx, max_score, predict_lbl, tmp_a, tmp_b, count, mean_a, mean_b, var_a, var_b, discr)
#endif
    for (int gene_a = 0; gene_a < Nprobe; ++gene_a) // for each gene
    {
      idx_sort_single[gene_a] = gene_a;

      // single gene case
      for (int i = 0; i < Nsample; ++i) // looCV cycle
      {
        max_score   = -inf;
        predict_lbl = -1;
        for (const auto & cl : member_class)
        {
          tmp_a   = (labels[i] == cl.first) ? data[gene_a][i] : 0.f;
          count   = (labels[i] == cl.first) ? static_cast < int >(cl.second.size()) - 1 : static_cast < int >(cl.second.size());
          mean_a  = (means[gene_a][cl.first] - tmp_a) / count;
          var_a   = static_cast < float >(count) / ((means_sq[gene_a][cl.first] - tmp_a * tmp_a) - mean_a * mean_a * count) + epsilon;
          discr   = - (data[gene_a][i] - mean_a) * var_a * (data[gene_a][i] - mean_a) // Mahalobis distance
                    // Uncomment for real diagQDA classifier
                    //- std :: log(var_a)
                    //+ std :: log(static_cast < float >(count) / static_cast < float >(cl.second.size()))
                    ;
          predict_lbl = (max_score < discr) ? cl.first : predict_lbl;
          max_score   = (max_score < discr) ? discr : max_score;
        }
        single_gene.class_score[predict_lbl][gene_a] += static_cast < int >(labels[i] == predict_lbl);
      } // end sample loop
      // update total and gene number
      single_gene.gene_a[gene_a]  = gene_a;
      single_gene.gene_b[gene_a]  = gene_a;
      single_gene.tot[gene_a]   = std::accumulate(  single_gene.class_score.get(), single_gene.class_score.get() + single_gene.n_class,
                                                    0, [&gene_a](const int & res, std :: unique_ptr < int[] > & score)
                                                    {
                                                      return res + score[gene_a];
                                                    });
      single_gene.mcc[gene_a]   = score :: matthews_corrcoef(single_gene.class_score[0][gene_a], static_cast < int >(member_class[0].size()), single_gene.class_score[1][gene_a], static_cast < int >(member_class[1].size()));

    } // end first gene loop

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
        std :: cout << "Sorting single gene..." << std :: flush;
        start_time = std :: chrono :: high_resolution_clock :: now();
      }
#ifdef _OPENMP
    }
#endif


#ifdef _OPENMP

#pragma omp single
    {
      mergeargsort_parallel_omp(idx_sort_single.get(), single_gene.tot.get(), 0, size_single, nth, [&](const int &a1, const int &a2){return single_gene.tot[a1] > single_gene.tot[a2];});
      if (diff_size_single)
      {
        std :: sort(idx_sort_single.get() + size_single, idx_sort_single.get() + Nprobe, [&](const int &a1, const int &a2){return single_gene.tot[a1] > single_gene.tot[a2];});
        std :: inplace_merge(idx_sort_single.get(), idx_sort_single.get() + size_single, idx_sort_single.get() + Nprobe, [&](const int &a1, const int &a2){return single_gene.tot[a1] > single_gene.tot[a2];});
      }
    }

#else

    std :: sort(idx_sort_single.get(), idx_sort_single.get() + Nprobe, [&](const int &a1, const int &a2){return single_gene.tot[a1] > single_gene.tot[a2];});

#endif
*/
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
      mergeargsort_parallel_omp(idx_sort_couples.get(), couples.tot.get(), 0, size_couples, nth, [&](const int &a1, const int &a2){return couples.tot[a1] > couples.tot[a2];});
      if (diff_size_couples)
      {
        std :: sort(idx_sort_couples.get() + size_couples, idx_sort_couples.get() + Ncomb, [&](const int &a1, const int &a2){return couples.tot[a1] > couples.tot[a2];});
        std :: inplace_merge(idx_sort_couples.get(), idx_sort_couples.get() + size_couples, idx_sort_couples.get() + Ncomb, [&](const int &a1, const int &a2){return couples.tot[a1] > couples.tot[a2];});
      }
    }
#else

    std :: sort(idx_sort_couples.get(), idx_sort_couples.get() + Ncomb, [&](const int &a1, const int &a2){return couples.tot[a1] > couples.tot[a2];});

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
    for (int i = 0; i < static_cast < int >(Ncomb * percentage); ++i)
    {
      const int idx = idx_sort_couples[i];
      results.tot[i] = couples.tot[idx];
      results.gene_a[i] = couples.gene_a[idx];
      results.gene_b[i] = couples.gene_b[idx];
      results.mcc[i] = couples.mcc[idx];

      for (int j = 0; j < Nclass; ++j)
        results.class_score[j][i] = couples.class_score[j][idx];
        //std :: swap(couples.class_score[j][i], couples.class_score[j][idx]);
      //std :: swap(couples.tot[i], couples.tot[idx]);
      //std :: swap(couples.gene_a[i], couples.gene_a[idx]);
      //std :: swap(couples.gene_b[i], couples.gene_b[idx]);
      //std :: swap(couples.mcc[i], couples.mcc[idx]);
    }
/*
#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < static_cast < int >(Nprobe * percentage); ++i)
    {
      const int idx = idx_sort_single[i];
      results.tot[i] = single_gene.tot[idx];
      results.gene_a[i] = single_gene.gene_a[idx];
      results.gene_b[i] = single_gene.gene_b[idx];
      results.mcc[i] = single_gene.mcc[idx];

      for (int j = 0; j < Nclass; ++j)
        results.class_score[j][i] = single_gene.class_score[j][idx];
        //std :: swap(single_gene.class_score[j][i], single_gene.class_score[j][idx]);

      //std :: swap(single_gene.tot[i], single_gene.tot[idx]);
      //std :: swap(single_gene.gene_a[i], single_gene.gene_a[idx]);
      //std :: swap(single_gene.gene_b[i], single_gene.gene_b[idx]);
      //std :: swap(single_gene.mcc[i], single_gene.mcc[idx]);
    }
*/
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

  return results;

} // end function
