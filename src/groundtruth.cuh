#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

// for convenience
using json = nlohmann::json;
using namespace std;

void compare_mathematica(vector<pair<int, int>> overlaps, const char *mmaFile);
void compare_mathematica(vector<pair<int, int>> overlaps,
                         const vector<int> &result_list, const char *mmaFile);