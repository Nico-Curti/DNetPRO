#ifndef __score_h__
#define __score_h__

#include <memory>  // std :: unique_ptr
#include <utility> // std :: move
#include <cmath>   // std :: sqrt

#include <utils.hpp> // old gcc compatibility


/**
* @class score
* @brief Abstract type representing the ensamble of usefull information to store during the
* couple evaluation.
*
* @details In particular the class includes:
*   - The Matthews Correlation Coefficient values of each couple (`mcc`)
*   - The first gene index of the couple (`gene_a`)
*   - The second gene index of the couple (`gene_b`)
*   - The total accuracy of the couple (`tot`)
*   - The accuracy score for each class (`class_score`)
*
*/
struct score
{

  std :: unique_ptr < float[] > mcc;  ///< Matthews Correlation Correlation of the couples
  std :: unique_ptr < int[] > gene_a; ///< First index of the couples
  std :: unique_ptr < int[] > gene_b; ///< Second index of the couples
  std :: unique_ptr < int[] > tot;    ///< Total accuracy of the couples

  std :: unique_ptr < std :: unique_ptr < int[] >[] > class_score; ///< Accuracy score for each class

  int N;       ///< The number of couples
  int n_class; ///< The number of classes

  // Constructors

  /**
  * @brief Default constructor.
  *
  */
  score ();
  /**
  * @brief Constructor with number of couples and number of classes
  *
  * @details This is the constructor used inside the DNetPRO algorithm in which
  * the number of couples can be evaluated as
  *
  * ```python
  *number_of_combination = number_of_samples * (number_of_samples - 1) / 2
  * ```
  *
  * @param N The number of available couples
  * @param n_class The number of available classes in which the samples are divided
  *
  */
  score (const int & N, const int & n_class);

  // Copy constructors

  /**
  * @brief Copy constructor.
  *
  * @details The operator doesn't perform a deep copy of the object but it just move all the buffers
  * from the input object to the current one. In this way we optimize the memory management.
  *
  * @param s Score object
  *
  */
  score (score & s);

  /**
  * @brief Copy operator.
  *
  * @details The operator doesn't perform a deep copy of the object but it just move all the buffers
  * from the input object to the current one. In this way we optimize the memory management.
  *
  * @param s Score object
  *
  */
  score & operator = (score && s);

  // Destructors

  /**
  * @brief Destructor set as default.
  *
  */
  ~score () = default;

  // Static functions

  /**
  * @brief Compute the Matthews Correlation Coefficient from the class scores
  *
  * @param s0 True positive score
  * @param m0 Total of true positive
  * @param s1 False Negative score
  * @param m1 Total of false negative
  *
  * @note This function is useful in the current implementation since we can easily manage
  * the information used by the score object without recomputing new ones.
  *
  */
  static float matthews_corrcoef (const float & s0, const int & m0, const float & s1, const int & m1);

};

#endif // __score_h__
