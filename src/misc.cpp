
#include "misc.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;



json read_json(const std::string& filename)
{   
    json input_par;
    std::ifstream file(filename);


    if (!file.is_open())
    {
        throw std::runtime_error("Could not open input file:" + filename);
    }
    try 
    {
        file >> input_par;
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error parsing JSON:" + std::string(e.what()));
    }

    return input_par;
}

void save_lm_expansion(const std::map<std::pair<int, int>, int>& lm_to_block, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }
    
    for (const auto& entry : lm_to_block) {
        outFile << entry.first.first << " " << entry.first.second << " " << entry.second << "\n";
    }
    
    outFile.close();
}

void normalize_array(std::array<double,3>& vec)
{      
    double norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    
    for (int idx = 0; idx< vec.size(); ++idx)
    {
        vec[idx] /= norm;
    }
}

void cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result)
{
    result[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    result[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    result[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}