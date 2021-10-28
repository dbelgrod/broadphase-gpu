#include <gpubf/groundtruth.h>


// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
unsigned long cantor(unsigned long x, unsigned long y)
{
    return (x + y)*(x + y + 1) / 2 + y;
}


void compare_mathematica(vector<unsigned long> overlaps, const char* jsonPath)
{
    // Get from file
    ifstream in(jsonPath);
    if(in.fail()) 
    {
        printf("%s does not exist\n", jsonPath);
        return;
    }

    json j_vec = json::parse(in);
    
    set<unsigned long> truePositives = j_vec.get<std::set<unsigned long>>();

    // Transform data to cantor
    set<unsigned long> algoBroadPhase;
    for (size_t i=0; i < overlaps.size(); i+=2)
    {
        algoBroadPhase.emplace(cantor(overlaps[i], overlaps[i+1]));
    }
                                               
    // Get intersection of true positive
    vector<unsigned long> algotruePositives(truePositives.size());
    vector<unsigned long>::iterator it=std::set_intersection (
        truePositives.begin(), truePositives.end(), 
        algoBroadPhase.begin(), algoBroadPhase.end(), algotruePositives.begin());
    algotruePositives.resize(it-algotruePositives.begin());
    
    printf("Contains %lu/%lu TP\n", algotruePositives.size(), truePositives.size());
    return;
}