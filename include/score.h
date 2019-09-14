#ifndef __score_h__
#define __score_h__

#include <memory>  // std :: unique_ptr
#include <utility> // std :: move
#include <cmath>   // std :: sqrt

#include <utility.hpp> // old gcc compatibility

struct score
{

  std :: unique_ptr < float[] > mcc;
  std :: unique_ptr < int[] > gene_a;
  std :: unique_ptr < int[] > gene_b;
  std :: unique_ptr < int[] > tot;

  std :: unique_ptr < std :: unique_ptr < int[] >[] > class_score;

  int N;
  int n_class;

  // Constructors

  score ();
  score (const int & N, const int & n_class);

  // Copy constructors

  score (score & s);
  score & operator = (score && s);

  // Destructors

  ~score () = default;

  // Static functions

  static float matthews_corrcoef (const float & s0, const int & m0, const float & s1, const int & m1);

};

#endif // __score_h__
