#pragma once
#include <string>
#include <filesystem>
#include "geoproc.hpp"

#if __has_include("Spectra/SymEigsSolver.h")
#define SPECTRA_INCLUDE
#include "Spectra/SymEigsSolver.h"
#include "Spectra/GenEigsSolver.h"
#include "Spectra/MatOp/SparseSymMatProd.h"
#include "Spectra/MatOp/SparseGenMatProd.h"
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>
#endif

#if __has_include("MatlabEngine.hpp")
#define MATLAB_ENGINE_INCLUDE
#pragma comment(lib, "libMatlabEngine.lib")
#pragma comment(lib, "libMatlabDataArray.lib")
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
using namespace matlab::engine;
#endif

namespace gp {
	#ifdef SPECTRA_INCLUDE
	void calcSpectraEigs(Eigen::SparseMatrix<double>& lbo, Eigen::VectorXd& eigenvals, Eigen::MatrixXd& eigenvecs,
								const uint32_t numeigs) {
		Spectra::SparseSymMatProd<double> op(lbo);
		Spectra::SymEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double>> eigs(&op, numeigs, 2 * numeigs);

		eigs.init();
		eigs.compute();

		eigenvals = eigs.eigenvalues().real();
		eigenvecs = eigs.eigenvectors().real();
	}

	void calcSpectraGenEigs(Eigen::SparseMatrix<double>& lbo, Eigen::SparseMatrix<double>& area, Eigen::VectorXd& eigenvals, Eigen::MatrixXd& eigenvecs,
									 const uint32_t numeigs) {
		Spectra::SparseSymMatProd<double> op(lbo);
		Spectra::SparseCholesky<double> bop(area);
		Spectra::SymGEigsSolver<double,
			Spectra::SMALLEST_MAGN,
			Spectra::SparseSymMatProd<double>,
			Spectra::SparseCholesky<double>,
			Spectra::GEIGS_MODE::GEIGS_CHOLESKY>
			eigs(&op, &bop, numeigs, 2 * numeigs);
		eigs.init();
		int nc = eigs.compute();

		eigenvals = eigs.eigenvalues();
		eigenvecs = eigs.eigenvectors();
	}
	#endif

	#ifdef MATLAB_ENGINE_INCLUDE
	auto calcMatlabGenEigs(const std::unique_ptr<MATLABEngine>& matlabPtr, const Eigen::SparseMatrix<double>& lb, const Eigen::SparseMatrix<double>& area, const size_t numeigs) {
		auto [lb_rows, lb_cols, lb_data] = getDataFromSparseMat(lb);
		auto [area_rows, area_cols, area_data] = getDataFromSparseMat(area);
		
		size_t lb_data_size = lb_data.size();
		size_t area_data_size = area_data.size();

		matlab::data::ArrayFactory factory;
		auto lb_data_p = factory.createBuffer<double>(lb_data_size);
		auto lb_rows_p = factory.createBuffer<size_t>(lb_data_size);
		auto lb_cols_p = factory.createBuffer<size_t>(lb_data_size);
		auto area_data_p = factory.createBuffer<double>(area_data_size);
		auto area_rows_p = factory.createBuffer<size_t>(area_data_size);
		auto area_cols_p = factory.createBuffer<size_t>(area_data_size);

		memcpy(lb_data_p.get(), lb_data.data(), sizeof(double) * lb_data_size);
		memcpy(lb_rows_p.get(), lb_rows.data(), sizeof(size_t) * lb_data_size);
		memcpy(lb_cols_p.get(), lb_cols.data(), sizeof(size_t) * lb_data_size);
		memcpy(area_data_p.get(), area_data.data(), sizeof(double) * area_data_size);
		memcpy(area_rows_p.get(), area_rows.data(), sizeof(size_t) * area_data_size);
		memcpy(area_cols_p.get(), area_cols.data(), sizeof(size_t) * area_data_size);

		matlab::data::SparseArray<double> A =
			factory.createSparseArray<double>({ (unsigned long long)lb.rows(), (unsigned long long)lb.cols() }, lb_data_size,
																				std::move(lb_data_p),
																				std::move(lb_rows_p),
																				std::move(lb_cols_p));
		matlab::data::SparseArray<double> B =
			factory.createSparseArray<double>({ (unsigned long long)lb.rows(), (unsigned long long)lb.cols() }, area_data_size,
																				std::move(area_data_p),
																				std::move(area_rows_p),
																				std::move(area_cols_p));

		std::vector<matlab::data::Array> args({ std::move(A), std::move(B),
																		 factory.createScalar<int32_t>(numeigs), factory.createCharArray("smallestabs") });
		auto result = matlabPtr->feval("eigs", 2, args);

		auto phi = result[0];
		auto lambda = result[1];

		Eigen::MatrixXd eigenvecs(lb.rows(), numeigs);
		for (size_t t = 0; t < lb.rows(); t++) {
			for (size_t i = 0; i < numeigs; i++) {
				eigenvecs(t, i) = (double)phi[t][i];
			}
		}

		Eigen::VectorXd eigenvals(numeigs);
		for (size_t i = 0; i < numeigs; i++) { eigenvals[i] = double(lambda[i][i]); }

		return std::pair(std::move(eigenvals), std::move(eigenvecs));
	}

	auto calcMatlabGenEigs(const Eigen::SparseMatrix<double>& lb, const Eigen::SparseMatrix<double>& area, const size_t numeigs) {
		std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();;
		return calcMatlabGenEigs(matlabPtr, lb, area, numeigs);
	}

	void saveEigen(const Eigen::VectorXd& evals, const Eigen::MatrixXd& evecs, const std::string& name) {
		std::ofstream output("./eigen_data/" + name + "_evals.txt");
		for (size_t i = 0; i < evals.size(); i++) {
			output << evals(i) << "\n";
		}
		output.close();
		output.open("./eigen_data/" + name + "_evecs.txt");
		for (size_t i = 0; i < evecs.rows(); i++) {
			for (size_t t = 0; t < evecs.cols() - 1; t++) {
				output << evecs(i, t) << ',';
			}
			output << evecs(i, evecs.cols() - 1) << "\n";
		}
		output.close();
	}

	auto loadEigen(const size_t n, const size_t numeigs, const std::string& name) {
		Eigen::MatrixXd eigenvecs(n, numeigs);
		Eigen::VectorXd eigenvals(numeigs);

		std::ifstream input("./eigen_data/" + name + "_evals.txt");
		std::string line;
		
		for (size_t i = 0; i < numeigs; i++) {
			getline(input, line);
			eigenvals(i) = abs(stod(line));
		}
		input.close();

		input.open("./eigen_data/" + name + "_evecs.txt");
		for (size_t i = 0; i < n; i++) {
			getline(input, line);
			auto sline = mio::split(line, ',');
			for (size_t t = 0; t < numeigs; t++) {
				eigenvecs(i, t) = stod(sline[t]);
			}
		}
		input.close();

		return std::pair(std::move(eigenvals), std::move(eigenvecs));
	}
	#endif
}