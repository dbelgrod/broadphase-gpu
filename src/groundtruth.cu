#include <gpubf/groundtruth.cuh>

// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
unsigned long cantor(unsigned long x, unsigned long y) {
  return (x + y) * (x + y + 1) / 2 + y;
}

struct cmp {
  bool operator()(pair<long, long> &a, pair<long, long> &b) const {
    return a.first == b.first && a.second == b.second;
  };
};

void compare_mathematica(vector<pair<int, int>> overlaps,
                         const char *jsonPath) {
  vector<int> result_list;
  result_list.resize(overlaps.size());
  fill(result_list.begin(), result_list.end(), true);
  compare_mathematica(overlaps, result_list, jsonPath);
}

void compare_mathematica(vector<pair<int, int>> overlaps,
                         const vector<int> &result_list, const char *jsonPath) {
  // Get from file
  ifstream in(jsonPath);
  if (in.fail()) {
    printf("%s does not exist\n", jsonPath);
    return;
  } else
    printf("Comparing mathematica file %s\n", jsonPath);

  json j_vec = json::parse(in);

  set<pair<long, long>> truePositives;
  vector<array<long, 2>> tmp = j_vec.get<vector<array<long, 2>>>();
  for (auto &arr : tmp)
    truePositives.emplace(arr[0], arr[1]);

  set<pair<long, long>> algoBroadPhase;
  for (size_t i = 0; i < overlaps.size(); i += 1) {
    if (result_list[i])
      algoBroadPhase.emplace(overlaps[i].first, overlaps[i].second);
  }

  // Get intersection of true positive
  vector<pair<long, long>> algotruePositives(truePositives.size());
  vector<pair<long, long>>::iterator it = std::set_intersection(
      truePositives.begin(), truePositives.end(), algoBroadPhase.begin(),
      algoBroadPhase.end(), algotruePositives.begin());
  algotruePositives.resize(it - algotruePositives.begin());

  printf("Contains %lu/%lu TP\n", algotruePositives.size(),
         truePositives.size());
  return;
}