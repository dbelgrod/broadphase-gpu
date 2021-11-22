#pragma once

#include <vector>
#include <set>
#include <fstream>
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;
using namespace std;

void compare_mathematica(vector<pair<int,int>> overlaps, const char* mmaFile);
void compare_mathematica(vector<pair<int,int>> overlaps, const vector<bool>& result_list, const char* mmaFile);