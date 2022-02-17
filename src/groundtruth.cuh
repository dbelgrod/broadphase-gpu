#pragma once

#include <vector>
#include <utility> // std::pair

namespace stq::gpu {

void compare_mathematica(std::vector<std::pair<int, int>> overlaps,
                         const char *mmaFile);
void compare_mathematica(std::vector<std::pair<int, int>> overlaps,
                         const std::vector<int> &result_list,
                         const char *mmaFile);

} // namespace stq::gpu