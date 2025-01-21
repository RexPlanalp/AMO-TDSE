#pragma once 

#include <map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>



nlohmann::json read_json(const std::string& filename);
void save_lm_expansion(const std::map<std::pair<int, int>, int>& lm_to_block, const std::string& filename);
void normalize_array(std::array<double,3>& vec);
void cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result);