#include <utility.hpp>

void file_error (const std :: string & input)
{
  std :: cout << "Couples Error! Invalid database filename. Given: "
              << input
              << std :: endl;
  std :: exit (error_file);
}


std :: unique_ptr < int[] > lbl2num (const std :: vector < std :: string > & lbl)
{
  std :: unique_ptr < int[] > num (new int[lbl.size()]);
  std :: unordered_set < std :: string > str_lbl(lbl.begin(), lbl.end(), lbl.size() * sizeof (std :: string));
  std :: transform(lbl.begin(), lbl.end(), num.get(),
                   [&](const std :: string & l)
                   {
                     return std :: distance(str_lbl.begin(), str_lbl.find(l));
                   });
  return num;
}
