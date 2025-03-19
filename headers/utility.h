#pragma once


class simulation;
enum class RunMode;

#include <petscsys.h>

#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"
#include <map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <complex>


nlohmann::json read_json(const std::string& filename);
void create_directory(int rank, const std::string& directory);


void save_lm_expansion(const std::map<std::pair<int, int>, int>& lm_to_block, const std::string& filename);
void normalize_array(std::array<double,3>& vec);
void cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result);

void pes_pointwise_mult(const std::vector<double>& vec1, const std::vector<std::complex<double>>& vec2,std::vector<std::complex<double>>& result);
void pes_pointwise_add(const std::vector<std::complex<double>>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result);
void pes_pointwise_magsq(const std::vector<std::complex<double>>& vec, std::vector<std::complex<double>>& result);
std::complex<double> pes_simpsons_method(const std::vector<std::complex<double>>& vec,double dr);

std::complex<double> H(std::complex<double> r);
std::complex<double> He(std::complex<double> r);
void scale_vector(std::vector<double>& vec, double scale);

void checkErr(PetscErrorCode ierr, const char* msg);

PetscMatrix KroneckerProduct(const PetscMatrix& A, const PetscMatrix& B);

double f(int l, int m);

double g(int l, int m);

double a(int l, int m);

double atilde(int l, int m);

double b(int l, int m);

double btilde(int l, int m);

double d(int l, int m);

double dtilde(int l, int m);

double c(int l, int m);

double ctilde(int l, int m);

double alpha(int l, int m);

double beta(int l, int m);

double charlie(int l, int m);

double delta(int l, int m);

double echo(int l, int m);

double foxtrot(int l, int m);

void project_out_bound(const PetscMatrix& S, PetscVector& state, const simulation& sim);

std::complex<double> compute_Ylm(int l, int m, double theta, double phi);
