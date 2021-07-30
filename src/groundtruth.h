#pragma once

#include <vector>
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;
using namespace std;

void compare_mathematica(vector<int>& overlaps, const char* mmaFile);