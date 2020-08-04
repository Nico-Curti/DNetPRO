#include <utils.hpp>

void file_error (const std :: string & input)
{
  std :: cout << "Couples Error! Invalid database filename. Given: "
              << input
              << std :: endl;
  std :: exit (error_file);
}


std :: vector < std :: string > split (const std :: string & txt, const std :: string & del)
{
  std :: vector < std :: string > token;

  std :: size_t pos = txt.find_first_of(del);
  std :: size_t start = 0;
  std :: size_t end = txt.size();

  while (pos != std :: string :: npos)
  {
    if (pos)
      token.push_back(txt.substr(start, pos));

    start += pos + 1;

    pos = txt.substr(start, end).find_first_of(del);
  }

  if (start != end)
    token.push_back(txt.substr(start, pos));

  return token;
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
