#pragma once

#include <petscsys.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <complex>
#include <vector>
#include <sys/stat.h>


void checkError(PetscErrorCode ierr, const char* msg = "");

void save_lm_expansion(const std::map<std::pair<int, int>, int>& lm_to_block, const std::string& filename);

void create_directory(int rank, const std::string& directory);

void normalize_array(std::array<double,3>& vec);

void cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result);

void pes_pointwise_mult(const std::vector<double>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result);

void pes_pointwise_add(const std::vector<std::complex<double>>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result);

void pes_pointwise_magsq(const std::vector<std::complex<double>>& vec, std::vector<std::complex<double>>& result);

std::complex<double> pes_simpsons_method(const std::vector<std::complex<double>>& vec,double dr);

double H(double r);

void scale_vector(std::vector<double>& vec, double scale);