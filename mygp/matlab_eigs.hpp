#pragma once
#if __has_include("MatlabEngine.hpp")
#define MATLAB_ENGINE_INCLUDE
#pragma comment(lib, "libMatlabEngine.lib")
#pragma comment(lib, "libMatlabDataArray.lib")
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
using namespace matlab::engine;
#endif

namespace gp::eigs {
	#ifdef MATLAB_ENGINE_INCLUDE
	// @return evals, evecs |V|xK
	auto calcMatlabGenEigs(const std::unique_ptr<MATLABEngine>& matlabPtr, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, const size_t numeigs) {
		auto [lboRows, lboCols, lboData] = gp::eig::getDataFromSparseMat(lbo);
		std::vector<Eigen::Index> areasRows(areas.rows()), areasCols(areas.rows());
		std::iota(areasRows.begin(), areasRows.end(), 0);
		std::iota(areasCols.begin(), areasCols.end(), 0);

		size_t lboDataSize = lboData.size();
		size_t areasDataSize = areas.size();

		matlab::data::ArrayFactory factory;
		auto lboData_p = factory.createBuffer<double>(lboDataSize);
		auto lboRows_p = factory.createBuffer<size_t>(lboDataSize);
		auto lboCols_p = factory.createBuffer<size_t>(lboDataSize);
		auto areasData_p = factory.createBuffer<double>(areasDataSize);
		auto areasRows_p = factory.createBuffer<size_t>(areasDataSize);
		auto areasCols_p = factory.createBuffer<size_t>(areasDataSize);

		memcpy(lboData_p.get(), lboData.data(), sizeof(double) * lboDataSize);
		memcpy(lboRows_p.get(), lboRows.data(), sizeof(size_t) * lboDataSize);
		memcpy(lboCols_p.get(), lboCols.data(), sizeof(size_t) * lboDataSize);
		memcpy(areasData_p.get(), areas.data(), sizeof(double) * areasDataSize);
		memcpy(areasRows_p.get(), areasRows.data(), sizeof(size_t) * areasDataSize);
		memcpy(areasCols_p.get(), areasCols.data(), sizeof(size_t) * areasDataSize);

		matlab::data::SparseArray<double> A =
			factory.createSparseArray<double>({ (unsigned long long)lbo.rows(), (unsigned long long)lbo.cols() }, lboDataSize,
																				std::move(lboData_p),
																				std::move(lboRows_p),
																				std::move(lboCols_p));
		matlab::data::SparseArray<double> B =
			factory.createSparseArray<double>({ (unsigned long long)lbo.rows(), (unsigned long long)lbo.cols() }, areasDataSize,
																				std::move(areasData_p),
																				std::move(areasRows_p),
																				std::move(areasCols_p));

		std::vector<matlab::data::Array> args({ std::move(A), std::move(B),
																		 factory.createScalar<int32_t>((int32_t)numeigs), factory.createCharArray("smallestabs") });
		auto result = matlabPtr->feval("eigs", 2, args);

		auto phi = result[0];
		auto lambda = result[1];

		Eigen::MatrixXd eigenvecs(lbo.rows(), numeigs);
		for (size_t t = 0; t < (size_t)lbo.rows(); t++) {
			for (size_t i = 0; i < numeigs; i++) {
				eigenvecs(t, i) = (double)phi[t][i];
			}
		}

		Eigen::VectorXd eigenvals(numeigs);
		for (size_t i = 0; i < numeigs; i++) { eigenvals[i] = (double)lambda[i][i]; }

		return std::pair(std::move(eigenvals), std::move(eigenvecs));
	}

	// @return evals, evecs |V|xK
	auto calcMatlabGenEigs(const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, const size_t numeigs) {
		std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
		return calcMatlabGenEigs(matlabPtr, lbo, areas, numeigs);
	}

	void saveEigen(const Eigen::VectorXd& evals, const Eigen::MatrixXd& evecs, const std::string& path) {
		std::ofstream output(path + "_evals.txt");
		for (size_t i = 0; i < (size_t)evals.size(); i++) {
			output << evals(i) << "\n";
		}
		output.close();
		output.open(path + "_evecs.txt");
		for (size_t i = 0; i <(size_t) evecs.rows(); i++) {
			for (size_t t = 0; t < (size_t)evecs.cols() - 1; t++) {
				output << evecs(i, t) << ',';
			}
			output << evecs(i, evecs.cols() - 1) << "\n";
		}
		output.close();
	}

	auto loadEigen(const size_t n, const size_t numeigs, const std::string& path) {
		Eigen::MatrixXd eigenvecs(n, numeigs);
		Eigen::VectorXd eigenvals(numeigs);

		std::ifstream input(path + "_evals.txt");
		std::string line;

		for (size_t i = 0; i < numeigs; i++) {
			getline(input, line);
			eigenvals(i) = abs(stod(line));
		}
		input.close();

		input.open(path + "_evecs.txt");
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