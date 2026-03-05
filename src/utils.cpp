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

  if ( txt.empty() )
    return token;

  std :: size_t start = 0;

  while (true)
  {
    // find the next del starting from 'start'
    std :: size_t pos = txt.find_first_of(del, start);

    if (pos == std :: string :: npos)
    {
      // last token (if not empty)
      if (start < txt.size())
        token.emplace_back(txt.substr(start));
      break;
    }

    if (pos > start)
    {
      // add no empty token found
      token.emplace_back(txt.substr(start, pos - start));
    }

    // restart from the next del found
    start = pos + 1;
  }

  return token;
}

std :: unique_ptr < int32_t[] > lbl2num (const std :: vector < std :: string > & lbl)
{
  std :: unique_ptr < int32_t[] > num(new int32_t[lbl.size()]);

  std :: unordered_map < std :: string, int32_t > map_lbl;
  map_lbl.reserve(lbl.size());

  int32_t next_id = 0;

  for (std :: size_t i = 0; i < lbl.size(); ++i)
  {
    const std :: string & l = lbl[i];

    auto it = map_lbl.find(l);
    if (it == map_lbl.end())
    {
      it = map_lbl.emplace(l, next_id).first;
      ++next_id;
    }

    num[i] = it->second;
  }
  return num;
}
