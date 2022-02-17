#include <stq/gpu/groundtruth.cuh>

#include <set>
#include <fstream>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
unsigned long cantor(unsigned long x, unsigned long y) {
  return (x + y) * (x + y + 1) / 2 + y;
}

struct cmp {
  bool operator()(std::pair<long, long> &a, std::pair<long, long> &b) const {
    return a.first == b.first && a.second == b.second;
  };
};

void compare_mathematica(std::vector<std::pair<int, int>> overlaps,
                         const char *jsonPath) {
  std::vector<int> result_list;
  result_list.resize(overlaps.size());
  fill(result_list.begin(), result_list.end(), true);
  compare_mathematica(overlaps, result_list, jsonPath);
}

void compare_mathematica(std::vector<std::pair<int, int>> overlaps,
                         const std::vector<int> &result_list,
                         const char *jsonPath) {
  // Get from file
  std::ifstream in(jsonPath);
  if (in.fail()) {
    spdlog::trace("{:s} does not exist", jsonPath);
    return;
  } else
    spdlog::trace("Comparing mathematica file {:s}", jsonPath);

  json j_vec = json::parse(in);

  std::set<std::pair<long, long>> truePositives;
  std::vector<std::array<long, 2>> tmp =
    j_vec.get<std::vector<std::array<long, 2>>>();
  for (auto &arr : tmp)
    truePositives.emplace(arr[0], arr[1]);

  std::set<std::pair<long, long>> algoBroadPhase;
  for (size_t i = 0; i < overlaps.size(); i += 1) {
    if (result_list[i])
      algoBroadPhase.emplace(overlaps[i].first, overlaps[i].second);
  }

  // Get intersection of true positive
  std::vector<std::pair<long, long>> algotruePositives(truePositives.size());
  std::vector<std::pair<long, long>>::iterator it = std::set_intersection(
    truePositives.begin(), truePositives.end(), algoBroadPhase.begin(),
    algoBroadPhase.end(), algotruePositives.begin());
  algotruePositives.resize(it - algotruePositives.begin());

  spdlog::trace("Contains {:d}/{:d} TP", algotruePositives.size(),
                truePositives.size());
  return;
}