#include <score.h>

score :: score () : mcc (nullptr), gene_a (nullptr), gene_b (nullptr), tot (nullptr), class_score (nullptr), N (0), n_class (0)
{
}

score :: score (const int32_t & N, const int32_t & n_class) : N (N), n_class (n_class)
{
  this->tot = std :: make_unique < int32_t[] >(N);
  this->gene_a = std :: make_unique < int32_t[] >(N);
  this->gene_b = std :: make_unique < int32_t[] >(N);

  this->mcc = std :: make_unique < float[] >(N);
  this->class_score = std :: make_unique < std :: unique_ptr < int32_t[] >[] >(n_class);

  for (int32_t i = 0; i < n_class; ++i)
  {
    this->class_score[i] = std :: make_unique < int32_t[] >(N);
    std :: fill_n(this->class_score[i].get(), N, 0);
  }
}

score :: score (score & s) : N (s.N), n_class (s.n_class)
{
  this->tot = std :: move(s.tot);
  this->gene_a = std :: move(s.gene_a);
  this->gene_b = std :: move(s.gene_b);
  this->mcc = std :: move(s.mcc);

  this->class_score = std :: move(s.class_score);
}

score & score :: operator = (score && s)
{
  this->N = s.N;
  this->n_class = s.n_class;
  this->tot = std :: move(s.tot);
  this->gene_a = std :: move(s.gene_a);
  this->gene_b = std :: move(s.gene_b);
  this->mcc = std :: move(s.mcc);

  this->class_score = std :: move(s.class_score);

  return *this;
}

float score :: matthews_corrcoef (const float & s0, const int32_t & m0, const float & s1, const int32_t & m1)
{
  const float num = (m0 * s1 - m1 * (m0 - s0));
  const float den = std :: sqrt(m0 * m1 * (s0 + m1 - s1) * (s1 + m0 - s0));
  return num / den;
}
