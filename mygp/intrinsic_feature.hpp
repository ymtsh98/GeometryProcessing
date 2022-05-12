#pragma once
#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/geocore.hpp>

//#include <unsupported/Eigen/FFT>

namespace gp::feat {
	template <typename T>
	Eigen::VectorXd meanCurvature(const heds::Mesh<T>& mesh, const Eigen::SparseMatrix<double>& lbo,
																const Eigen::VectorXd& areas) {

		Eigen::MatrixXd verts(mesh.getNumVerts(), 3);
		for (size_t i = 0; i < mesh.getNumVerts(); i++) verts.row(i) = mesh.getPosition<Eigen::Vector3d>(i);

		Eigen::SparseMatrix<double> A(mesh.getNumVerts(), mesh.getNumVerts());
		A.setIdentity();
		A.diagonal().array() *= (1.0 / areas.array());

		return 0.5 * (0.5 * A * lbo * verts).rowwise().norm();
	}

	template <typename T>
	Eigen::VectorXd gaussianCurvature(const heds::Mesh<T>& mesh, const Eigen::VectorXd& areas) {
		Eigen::VectorXd curvature(mesh.getNumVerts());
		curvature.fill(2.0 * std::numbers::pi);
	
		for (const auto& face : mesh.faces) {
			auto v1 = face->edge->vert->id;
			auto v2 = face->edge->next->vert->id;
			auto v3 = face->edge->next->next->vert->id;
	
			Eigen::Vector3d l12 = mesh.getPosition<Eigen::Vector3d>(v2) - mesh.getPosition<Eigen::Vector3d>(v1);
			Eigen::Vector3d l23 = mesh.getPosition<Eigen::Vector3d>(v3) - mesh.getPosition<Eigen::Vector3d>(v2);
			Eigen::Vector3d l31 = mesh.getPosition<Eigen::Vector3d>(v1) - mesh.getPosition<Eigen::Vector3d>(v3);
	
			double theta1 = acos(-l12.dot(l31)  / (l12.norm() * l31.norm()));
			double theta2 = acos(-l23.dot(l12)  / (l23.norm() * l12.norm()));
			double theta3 = acos(-l31.dot(l23)  / (l31.norm() * l23.norm()));
	
			curvature[v1] -= theta1;
			curvature[v2] -= theta2;
			curvature[v3] -= theta3;
		}
	
		curvature.array() /= areas.array();
		return curvature;
	}
	
	Eigen::VectorXd hksParams(const double minEigVal, const double maxEigVal, const size_t numSample) {
		Eigen::VectorXd params(numSample);
		double minParam = abs(4.0 * log(10.0) / maxEigVal);
		double maxParam = abs(4.0 * log(10.0) / minEigVal);
		double step = log(maxParam) - log(minParam);
	
		for (size_t i = 0; i < numSample; i++) {
			params[i] = exp(log(minParam) + step * (i + 1) / (double)numSample);
		}
	
		return params;
	}
	
	Eigen::VectorXd heatKernelSignature(const Eigen::VectorXd& eigenvals,
																			const Eigen::MatrixXd& eigenvecs,
																			const double time, const int y = -1) {
	
		Eigen::VectorXd heat = Eigen::VectorXd::Zero(eigenvecs.rows());
		size_t numeigs = eigenvals.size();
		double scale = 0.0;
	
		if (y == -1) {
			for (size_t i = 1; i < numeigs; i++) {
				scale += exp(-eigenvals[i] * time);
				heat = heat.array() + exp(-eigenvals[i] * time) * Eigen::pow(eigenvecs.col(i).array(), 2);
			}
			heat.array() /= scale;
	
		} else {
			for (size_t i = 1; i < numeigs; i++) {
				scale += exp(-eigenvals[i] * time);
				for (size_t x = 0; x < (size_t)eigenvecs.rows(); x++) {
					heat[x] += exp(-eigenvals[i] * time) * eigenvecs(x, i) * eigenvecs(y, i);
				}
			}
			heat.array() /= scale;
		}
	
		return heat;
	}
	
	std::pair<Eigen::VectorXd, double> wksParams(const double minEigVal, const double maxEigVal, const size_t numSample) {
		Eigen::VectorXd params(numSample);
		double minParam = log(maxEigVal) / 1.02;
		double maxParam = log(minEigVal);
		double step = maxParam - minParam;
	
		for (size_t i = 0; i < numSample; i++) {
			params[i] = minParam + step * (i + 1) / (double)numSample;
		}
	
		double sigma = numSample > 0 ? (params[1] - params[0]) * 5.0 : 0.0;
	
		return { params, sigma };
	}
	
	Eigen::VectorXd waveKernelSignature(const Eigen::VectorXd& eigenvals,
																			const Eigen::MatrixXd& eigenvecs,
																			const double energy, const double sigma) {
	
		Eigen::VectorXd wave = Eigen::VectorXd::Zero(eigenvecs.rows());
		size_t numeigs = eigenvals.size();
	
		double scale = 0.0;
		for (size_t i = 1; i < numeigs; i++) {
			scale += exp(-pow(energy - log(eigenvals(i)), 2) / (2.0 * sigma * sigma));
			double c = exp(-pow(energy - log(eigenvals(i)), 2) / (2.0 * sigma * sigma));
			wave.array() += Eigen::pow(eigenvecs.col(i).array(), 2) * c;
		}
	
		wave.array() /= scale;
	
		return wave;
	}
	
	// heat kernel signatures
	// @return |V| x numSample matrix
	Eigen::MatrixXd heatKernelSigs(const Eigen::VectorXd& eigenvals,
															const Eigen::MatrixXd& eigenvecs,
															const size_t numSample) {
		Eigen::MatrixXd hks(eigenvecs.rows(), numSample);
	
		size_t numeigs = eigenvals.size();
		auto energy = hksParams(eigenvals[1], eigenvals[numeigs - 1], numSample);
	
		Eigen::MatrixXd phi2 = Eigen::pow(eigenvecs.block(0, 1, eigenvecs.rows(), numeigs - 1).array(), 2.0);
		Eigen::VectorXd E = eigenvals.block(1, 0, numeigs - 1, 1);
		for (size_t i = 0; i < numSample; i++) {
			Eigen::VectorXd C = Eigen::exp(-energy[i] * E.array());
	
			hks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}
	
		return hks;
	}
	
	// heat kernel signatures
	// @return |V| x numSample matrix
	Eigen::MatrixXd heatKernelSigs(const Eigen::VectorXd& eigenvals,
															const Eigen::MatrixXd& eigenvecs,
															const size_t numSample, const size_t landmark) {
		Eigen::MatrixXd hks(eigenvecs.rows(), numSample);
	
		size_t numeigs = eigenvals.size();
		//auto energy = hksParams(eigenvals[1], eigenvals[numeigs - 1], numSample);
		auto energy = hksParams(eigenvals[1], eigenvals.maxCoeff(), numSample);
	
		Eigen::MatrixXd lm = eigenvecs.row(landmark).replicate(eigenvecs.rows(), 1);
		Eigen::MatrixXd phi2 = (lm.array() * eigenvecs.array()).block(0, 1, eigenvecs.rows(), numeigs - 1);

		Eigen::VectorXd E = eigenvals.block(1, 0, numeigs - 1, 1);
		for (size_t i = 0; i < numSample; i++) {
			Eigen::VectorXd C = Eigen::exp(-energy[i] * E.array());
	
			hks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}
	
		return hks;
	}
	
	//// scale-invariant heat kernel signature
	//// @return |V| x numSample matrix
	//Eigen::MatrixXd timeStepSIHKS(const Eigen::VectorXd& eigenvals,
	//														const Eigen::MatrixXd& eigenvecs,
	//														const size_t numSample, 
	//														const double logBasis=1.5) {
	//	Eigen::MatrixXd hks(eigenvecs.rows(), numSample);
	//
	//	size_t numeigs = eigenvals.size();
	//	auto energy = hksParams(eigenvals[1], eigenvals[numeigs - 1], numSample);
	//	for (size_t i = 0; i < numSample; i++) energy(i) = i * 0.2 + 1.0;
	//
	//	Eigen::MatrixXd phi2 = Eigen::pow(eigenvecs.block(0, 1, eigenvecs.rows(), numeigs - 1).array(), 2.0);
	//	Eigen::VectorXd E = eigenvals.block(1, 0, numeigs - 1, 1);
	//	for (size_t i = 0; i < numSample; i++) {
	//		Eigen::VectorXd A = pow(logBasis, energy[i]) * E;
	//		Eigen::VectorXd C = Eigen::exp(-A.array());
	//
	//		hks.col(i).array() = 
	//			- log(logBasis) * 
	//			(phi2.array().rowwise() * 
	//			 (C.array() * 
	//				A.array()).transpose()).rowwise().sum() /
	//			(phi2.array().rowwise() * C.transpose().array()).rowwise().sum();
	//	}
	//
	//	Eigen::FFT<double> fft;
	//	Eigen::MatrixXcd sihks(hks.rows(), hks.cols());
	//	for (size_t i = 0; i < numSample; i++) {
	//		auto ff = fft.fwd(hks.col(i));
	//		std::cout << ff.rows() << " " << ff.cols() << std::endl; 
	//		sihks.col(i) = fft.fwd(hks.col(i));
	//	}
	//	
	//	return sihks.cwiseAbs().block(0, 0, eigenvecs.rows(), 50);
	//	//return sihks.real();
	//
	//	//return hks.block(0, 0, eigenvecs.rows(), 50);
	//}

	// wave kernel signatures
	// @return |V| x numSample matrix
	Eigen::MatrixXd waveKernelSigs(const Eigen::VectorXd& eigenvals,
															const Eigen::MatrixXd& eigenvecs,
															const size_t numSample) {
	
		Eigen::MatrixXd wks(eigenvecs.rows(), numSample);
	
		size_t numeigs = eigenvals.size();
		//auto [energy, sigma] = wksParams(eigenvals[1], eigenvals[numeigs-1], numSample);
		auto [energy, sigma] = wksParams(eigenvals[1], eigenvals.maxCoeff(), numSample);
	
		Eigen::MatrixXd phi2 = Eigen::pow(eigenvecs.block(0, 1, eigenvecs.rows(), numeigs-1).array(), 2.0);
		Eigen::VectorXd E = Eigen::log(eigenvals.array()).block(1, 0, numeigs-1, 1);
	
		for (size_t i = 0; i < numSample; i++) {
			Eigen::VectorXd C = Eigen::exp(-Eigen::pow(energy[i] * Eigen::VectorXd::Ones(numeigs-1).array()
																								 - E.array(), 2.0) / (2.0 * sigma * sigma));
	
			wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}
	
		return wks;
	}
	
	// wave kernel signatures
	// @return |V| x numSample matrix
	Eigen::MatrixXd waveKernelSigs(const Eigen::VectorXd& eigenvals,
															const Eigen::MatrixXd& eigenvecs,
															const size_t numSample,
															const size_t landmark) {
		Eigen::MatrixXd wks(eigenvecs.rows(), numSample);
	
		size_t numeigs = eigenvals.size();
		//auto [energy, sigma] = wksParams(eigenvals[1], eigenvals[numeigs-1], numSample);
		auto [energy, sigma] = wksParams(eigenvals[1], eigenvals.maxCoeff(), numSample);
	
		Eigen::MatrixXd lm = eigenvecs.row(landmark).replicate(eigenvecs.rows(), 1);
		Eigen::MatrixXd phi2 = (lm.array() * eigenvecs.array()).block(0, 1, eigenvecs.rows(), numeigs - 1);
	
		Eigen::VectorXd E = Eigen::log(eigenvals.array()).block(1, 0, numeigs-1, 1);
	
		for (size_t i = 0; i < numSample; i++) {
			Eigen::VectorXd C = Eigen::exp(-Eigen::pow(energy[i] * Eigen::VectorXd::Ones(numeigs-1).array()
																								 - E.array(), 2.0) / (2.0 * sigma * sigma));
	
			wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}
	
		return wks;
	}
	
	// mean curvatures
	// @return |V| x numSample matrix
	template <typename T>
	Eigen::MatrixXd meanCurvatures(const heds::Mesh<T>& mesh, const Eigen::SparseMatrix<double>& lbo, 
														 const Eigen::VectorXd& areas, const size_t numSample, const double time = 0.2) {
		Eigen::MatrixXd mc(mesh.getNumVerts(), numSample);
		mc.col(0) = meanCurvature(mesh, lbo, areas);

		Eigen::SparseMatrix<double> A(lbo.rows(), lbo.rows());
		A.setIdentity();
		//A.diagonal().array() *= areas.array();
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(A + time * lbo);

		for (size_t i = 1; i < numSample; i++) {
			mc.col(i) = solver.solve(mc.col(i-1));
		}
		return mc;
	}
	
	// gaussian curvatures
	// @return |V| x numSample matrix
	template <typename T>
	Eigen::MatrixXd gaussianCurvatures(const heds::Mesh<T>& mesh, const Eigen::SparseMatrix<double>& lbo, 
														 const Eigen::VectorXd& areas, const size_t numSample, const double time = 0.2) {
		Eigen::MatrixXd gc(mesh.getNumVerts(), numSample);
		gc.col(0) = gaussianCurvature(mesh, areas);

		Eigen::SparseMatrix<double> A(lbo.rows(), lbo.rows());
		A.setIdentity();
		//A.diagonal().array() *= areas.array();
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(A + time * lbo);

		for (size_t i = 1; i < numSample; i++) {
			gc.col(i) = solver.solve(gc.col(i-1));
		}
		return gc;
	}
}