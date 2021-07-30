#pragma once

#include <vector>
#include <set>
#include <fstream>
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;
using namespace std;

// struct Ovrlap
// {
//     vector<unsigned long> ee;
//     vector<unsigned long> vf;
//     Ovrlap::size()
//     {
//         return ee.size() + vf.size();
//     }
//     unsigned long& Ovrlap::operator[](unsigned long i);
//     {
//         unsigned long ret = i < ee.size() ? ee[i] : vf[i - ee.size()];
//     }
// };

void compare_mathematica(vector<unsigned long> overlaps, const char* mmaFile);