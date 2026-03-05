#include <score.h>

score :: score () : mcc (nullptr), gene_a (nullptr), gene_b (nullptr), tot (nullptr), class_score (nullptr), confusion (nullptr), N (0), n_class (2)
{
}

score :: score (const int32_t & N, const int32_t & n_class) : N (N), n_class (n_class)
{
  this->tot = std :: make_unique < int32_t[] >(N);
  this->gene_a = std :: make_unique < int32_t[] >(N);
  this->gene_b = std :: make_unique < int32_t[] >(N);

  this->mcc = std :: make_unique < float[] >(N);

  const std :: size_t Nnclass = static_cast < std :: size_t >(N) * static_cast < std :: size_t >(n_class);

  this->class_score = std :: make_unique < int32_t[] >(Nnclass);
  this->confusion = std :: make_unique < uint32_t[] >(Nnclass * static_cast<size_t>(n_class));
  
  std :: fill_n(this->tot.get(), N, 0);
  std :: fill_n(this->gene_a.get(), N, 0);
  std :: fill_n(this->gene_b.get(), N, 0);
  std :: fill_n(this->mcc.get(), N, 0);
  std :: fill_n(this->class_score.get(), Nnclass, 0);
  std :: fill_n(this->confusion.get(), Nnclass * static_cast<size_t>(n_class), 0u);
}

float score :: matthews_corrcoef (
  const int32_t * row_sum,
  const int32_t * col_sum,
  const int32_t & K,
  const int32_t & trace,
  const int32_t & n)
{
  long long sum_row_col = 0;
  long long sum_row2 = 0;
  long long sum_col2 = 0;

  for (int32_t k = 0; k < K; ++k)
  {
    const long long r = row_sum[k];
    const long long c = col_sum[k];
    sum_row_col += r * c;
    sum_row2    += r * r;
    sum_col2    += c * c;
  }

  const long long nn  = (long long)n * (long long)n;
  const long long num = (long long)n * (long long)trace - sum_row_col;

  const long long dl = nn - sum_col2;
  const long long dr = nn - sum_row2;

  if (dl <= 0 || dr <= 0) return 0.f;

  const double den = std::sqrt((double)dl * (double)dr);
  return (float)((double)num / den);
}
