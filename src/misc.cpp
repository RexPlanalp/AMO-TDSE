
#include "misc.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <complex>
#include <vector>

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

    if (norm == 0) return;
    
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

void pes_pointwise_mult(const std::vector<double>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result) {
    result.resize(vec1.size());  // Correct: Set the size

    for (int idx = 0; idx < vec1.size(); ++idx) {
        result[idx] = vec1[idx] * vec2[idx]; // Direct assignment: No push_back!
    }
}

void pes_pointwise_add(const std::vector<std::complex<double>>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result) {
    result.resize(vec1.size());

    for (int idx = 0; idx < vec1.size(); ++idx) {
        result[idx] = vec1[idx] + vec2[idx]; // Direct assignment
    }
}

void pes_pointwise_magsq(const std::vector<std::complex<double>>& vec, std::vector<std::complex<double>>& result) {
    result.resize(vec.size());

    for (int idx = 0; idx < vec.size(); ++idx) {
        result[idx] = vec[idx] * std::conj(vec[idx]); // Direct assignment
    }
}

std::complex<double> pes_simpsons_method(const std::vector<std::complex<double>>& vec,double dr)
{
    int n = vec.size() - 1;
    std::complex<double> I {};


    if (n % 2 != 0)
    {
        I += (3 * dr / 8) * (vec[n-2]+static_cast<double>(3)*vec[n-1]+static_cast<double>(3)*vec[n]);
        n -= 2;
    }

    double p = dr / 3; 
    I += (vec[0]+vec[n]) * p;
    for (int vec_idx = 1; vec_idx < n; vec_idx += 2)
    {
        I += static_cast<double>(4) * vec[vec_idx] * p;
    }
    for (int vec_idx = 2; vec_idx < n-1; vec_idx += 2)
    {
        I += static_cast<double>(2) * vec[vec_idx] * p;
    }
    
    return I;

}
 