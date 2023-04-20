#pragma once
#include <mygp/geoutil.hpp>
#include <mygp/eigutil.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/laplacian.hpp>

//#include <unsupported/Eigen/FFT>

namespace gp::feat {

#ifndef MYGP_IMPLEMENTATION
Eigen::VectorXd meanCurvature(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, const Eigen::MatrixX3d& normals);

Eigen::VectorXd gaussianCurvature(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXd& areas);

Eigen::VectorXd hksParams(const double minEigVal, const double maxEigVal, const size_t numSample);

Eigen::VectorXd heatKernelSignature(const Eigen::VectorXd& eigenvals, const Eigen::MatrixXd& eigenvecs,	const double time, const int y = -1);

std::pair<Eigen::VectorXd, double> wksParams(const double minEigVal, const double maxEigVal, const size_t numSample);

Eigen::VectorXd waveKernelSignature(const Eigen::VectorXd& eigenvals,	const Eigen::MatrixXd& eigenvecs,	const double energy, const double sigma);

// heat kernel signatures
// @return |V| x numSample matrix
Eigen::MatrixXd heatKernelSigs(const Eigen::VectorXd& eigenvals, const Eigen::MatrixXd& eigenvecs, const size_t numSample);

// heat kernel signatures
// @return |V| x numSample matrix
Eigen::MatrixXd heatKernelSigs(const Eigen::VectorXd& eigenvals,	const Eigen::MatrixXd& eigenvecs,	const size_t numSample, const size_t landmark);

// wave kernel signatures
// @return |V| x numSample matrix
Eigen::MatrixXd waveKernelSigs(const Eigen::VectorXd& eigenvals, const Eigen::MatrixXd& eigenvecs, const size_t numSample);

// wave kernel signatures
// @return |V| x numSample matrix
Eigen::MatrixXd waveKernelSigs(const Eigen::VectorXd& eigenvals,	const Eigen::MatrixXd& eigenvecs,	const size_t numSample,	const size_t landmark);

// mean curvatures
// @return |V| x numSample matrix
Eigen::MatrixXd meanCurvatures(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::SparseMatrix<double>& lbo,
	const Eigen::VectorXd& areas, const Eigen::MatrixX3d& normals, const size_t numSample, const double time = 0.2);

// gaussian curvatures
// @return |V| x numSample matrix
Eigen::MatrixXd gaussianCurvatures(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::SparseMatrix<double>& lbo,
	const Eigen::VectorXd& areas, const size_t numSample, const double time = 0.2);

// wavelet heat kernel generator
class WaveletHeatKernel {
	std::reference_wrapper<const Eigen::SparseMatrix<double>> lbo;
	Eigen::SparseMatrix<double> A;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> msolver, tsolver;
	size_t sampleSize;

public:

	/// double srcScale = sqrt(srcData.areas.sum()), trgScale = sqrt(trgData.areas.sum());
	/// normFactor = srcScale
	/// timeStep = 1.0 / timeStep / trgScale
	WaveletHeatKernel(heds::Mesh<Eigen::MatrixX3d>& mesh,	const Eigen::SparseMatrix<double>& lbo,
		const double normFactor, const double timeStep,const size_t sampleSize);

	/// V x (sample size * source size)
	template <typename T = int>
	Eigen::MatrixXd compute(const std::vector<T>& source);

	/// V x sampleSize
	template <typename T = int>
	Eigen::MatrixXd computeFlat(const  std::vector<T>& source);

};

#else
#define MYGP_IMPLEMENTATION

inline Eigen::VectorXd meanCurvature(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::SparseMatrix<double>& lbo,
															const Eigen::VectorXd& areas, const Eigen::MatrixX3d& normals) {

	Eigen::VectorXd meanCurvature(const heds::Mesh<Eigen::MatrixX3d>&mesh, const Eigen::SparseMatrix<double>&lbo,
		const Eigen::VectorXd& areas, const Eigen::MatrixX3d& normals) {
		Eigen::SparseMatrix<double> A(mesh.getNumVerts(), mesh.getNumVerts());
		A.setIdentity();
		A.diagonal().array() *= (0.5 / areas.array());

		Eigen::MatrixXd Lp = A * lbo * mesh.copyPosition();
		return (Lp.array() * normals.array()).rowwise().sum();
		//return (Lp.array() * normals.array()).rowwise().sum().cwiseSign().array() * (Lp.rowwise().norm()).array();
	}
}

inline Eigen::VectorXd gaussianCurvature(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXd& areas) {
	Eigen::VectorXd curvature = Eigen::VectorXd::Constant(mesh.getNumVerts(), 2.0 * std::numbers::pi);
	
	for (const auto& face : mesh.faces) {
		auto v1 = face->edge->vert->id;
		auto v2 = face->edge->next->vert->id;
		auto v3 = face->edge->next->next->vert->id;
	
		Eigen::Vector3d l12 = mesh.position(v2) - mesh.position(v1);
		Eigen::Vector3d l23 = mesh.position(v3) - mesh.position(v2);
		Eigen::Vector3d l31 = mesh.position(v1) - mesh.position(v3);
	
		double theta1 = util::angle(l12, -l31);
		double theta2 = util::angle(l23, -l12);
		double theta3 = util::angle(l31, -l23);
	
		curvature[v1] -= theta1;
		curvature[v2] -= theta2;
		curvature[v3] -= theta3;
	}
	
	curvature.array() /= areas.array();
	return curvature;
}
	
inline Eigen::VectorXd hksParams(const double minEigVal, const double maxEigVal, const size_t numSample) {
	Eigen::VectorXd params(numSample);
	double minParam = abs(4.0 * log(10.0) / maxEigVal);
	double maxParam = abs(4.0 * log(10.0) / minEigVal);
	double step = log(maxParam) - log(minParam);
	
	for (size_t i = 0; i < numSample; i++) {
		params[i] = exp(log(minParam) + step * (i + 1) / (double)numSample);
	}
	
	return params;
}
	
inline Eigen::VectorXd heatKernelSignature(const Eigen::VectorXd& eigenvals,
																		const Eigen::MatrixXd& eigenvecs,
																		const double time, const int y = -1) {
	
	Eigen::VectorXd heat = Eigen::VectorXd::Zero(eigenvecs.rows());
	size_t numeigs = eigenvals.size();
	double scale = 0.0;
	
	if (y == -1) {
		for (size_t i = 1; i < numeigs; i++) {
			scale += exp(-eigenvals[i] * time);
			heat = heat.array() + exp(-eigenvals[i] * time) * eigenvecs.col(i).array().square();
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
	
inline std::pair<Eigen::VectorXd, double> wksParams(const double minEigVal, const double maxEigVal, const size_t numSample) {
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
	
inline Eigen::VectorXd waveKernelSignature(const Eigen::VectorXd& eigenvals,
																		const Eigen::MatrixXd& eigenvecs,
																		const double energy, const double sigma) {
	
	Eigen::VectorXd wave = Eigen::VectorXd::Zero(eigenvecs.rows());
	size_t numeigs = eigenvals.size();
	
	double scale = 0.0;
	for (size_t i = 1; i < numeigs; i++) {
		scale += exp(-pow(energy - log(eigenvals(i)), 2) / (2.0 * sigma * sigma));
		double c = exp(-pow(energy - log(eigenvals(i)), 2) / (2.0 * sigma * sigma));
		wave.array() += eigenvecs.col(i).array().square() * c;
	}
	
	wave.array() /= scale;
	
	return wave;
}
	
// heat kernel signatures
// @return |V| x numSample matrix
inline Eigen::MatrixXd heatKernelSigs(const Eigen::VectorXd& eigenvals,
														const Eigen::MatrixXd& eigenvecs,
														const size_t numSample) {
	Eigen::MatrixXd hks(eigenvecs.rows(), numSample);
	
	size_t numeigs = eigenvals.size();
	auto energy = hksParams(eigenvals[1], eigenvals[numeigs - 1], numSample);
	
	Eigen::MatrixXd phi2 = eigenvecs.block(0, 1, eigenvecs.rows(), numeigs - 1).array().square();
	Eigen::VectorXd E = eigenvals.block(1, 0, numeigs - 1, 1);
	for (size_t i = 0; i < numSample; i++) {
		Eigen::VectorXd C = Eigen::exp(-energy[i] * E.array());
	
		hks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
	}
	
	return hks;
}
	
// heat kernel signatures
// @return |V| x numSample matrix
inline Eigen::MatrixXd heatKernelSigs(const Eigen::VectorXd& eigenvals,
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
inline Eigen::MatrixXd waveKernelSigs(const Eigen::VectorXd& eigenvals,
														const Eigen::MatrixXd& eigenvecs,
														const size_t numSample) {
	
	Eigen::MatrixXd wks(eigenvecs.rows(), numSample);
	
	size_t numeigs = eigenvals.size();
	//auto [energy, sigma] = wksParams(eigenvals[1], eigenvals[numeigs-1], numSample);
	auto [energy, sigma] = wksParams(eigenvals[1], eigenvals.maxCoeff(), numSample);
	double s = 2.0 * sigma * sigma;

	Eigen::MatrixXd phi2 = eigenvecs.block(0, 1, eigenvecs.rows(), numeigs-1).array().square();
	Eigen::VectorXd E = Eigen::log(eigenvals.array()).block(1, 0, numeigs-1, 1);
		
	#pragma omp parallel for
	for (int i = 0; i < (int)numSample; i++) {
		Eigen::VectorXd C = Eigen::exp(-(Eigen::VectorXd::Constant(numeigs-1, energy[i]).array()
																								- E.array()).square() / s);

		wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
	}
	
	return wks;
}
	
// wave kernel signatures
// @return |V| x numSample matrix
inline Eigen::MatrixXd waveKernelSigs(const Eigen::VectorXd& eigenvals,
														const Eigen::MatrixXd& eigenvecs,
														const size_t numSample,
														const size_t landmark) {
	Eigen::MatrixXd wks(eigenvecs.rows(), numSample);
	
	size_t numeigs = eigenvals.size();
	//auto [energy, sigma] = wksParams(eigenvals[1], eigenvals[numeigs-1], numSample);
	auto [energy, sigma] = wksParams(eigenvals[1], eigenvals.maxCoeff(), numSample);
	double s = 2.0 * sigma * sigma;

	Eigen::MatrixXd lm = eigenvecs.row(landmark).replicate(eigenvecs.rows(), 1);
	Eigen::MatrixXd phi2 = (lm.array() * eigenvecs.array()).block(0, 1, eigenvecs.rows(), numeigs - 1);
	
	Eigen::VectorXd E = Eigen::log(eigenvals.array()).block(1, 0, numeigs-1, 1);
	
	#pragma omp parallel for
	for (int i = 0; i < (int)numSample; i++) {
		Eigen::VectorXd C = Eigen::exp(-(Eigen::VectorXd::Constant(numeigs-1, energy[i]).array()
			- E.array()).square() / s);

		wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
	}
	
	return wks;
}
	
// mean curvatures
// @return |V| x numSample matrix
inline Eigen::MatrixXd meanCurvatures(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::SparseMatrix<double>& lbo,
														const Eigen::VectorXd& areas, const Eigen::MatrixX3d& normals, const size_t numSample, const double time = 0.2) {
	Eigen::MatrixXd mc(mesh.getNumVerts(), numSample);
	mc.col(0) = meanCurvature(mesh, lbo, areas, normals);

	Eigen::SparseMatrix<double> A(lbo.rows(), lbo.rows());
	A.setIdentity();
	//A.diagonal().array() *= 1.0 / areas.array();
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A + time * lbo);

	for (size_t i = 1; i < numSample; i++) {
		mc.col(i) = solver.solve(mc.col(i-1));
	}
	return mc;
}
	
// gaussian curvatures
// @return |V| x numSample matrix
inline Eigen::MatrixXd gaussianCurvatures(const heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::SparseMatrix<double>& lbo,
														const Eigen::VectorXd& areas, const size_t numSample, const double time = 0.2) {
	Eigen::MatrixXd gc(mesh.getNumVerts(), numSample);
	gc.col(0) = gaussianCurvature(mesh, areas);

	Eigen::SparseMatrix<double> A(lbo.rows(), lbo.rows());
	A.setIdentity();
	//A.diagonal().array() *= areas.array();
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A + time * lbo);

	for (size_t i = 1; i < numSample; i++) {
		gc.col(i) = solver.solve(gc.col(i-1));
	}
	return gc;
}

class WaveletHeatKernel {
	std::reference_wrapper<const Eigen::SparseMatrix<double>> lbo;
	Eigen::SparseMatrix<double> A;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> msolver, tsolver;
	size_t sampleSize;

public:

	/// double srcScale = sqrt(srcData.areas.sum()), trgScale = sqrt(trgData.areas.sum());
	/// normFactor = srcScale
	/// timeStep = 1.0 / timeStep / trgScale
	WaveletHeatKernel(
		heds::Mesh<Eigen::MatrixX3d>& mesh,
		const Eigen::SparseMatrix<double>& lbo,
		const double normFactor,
		const double timeStep,
		const size_t sampleSize) :
		lbo(lbo),
		sampleSize(sampleSize) {
		auto refPos = mesh.getPbind();
		Eigen::MatrixX3d nV = refPos->array() / normFactor;
		mesh.bind(nV);
		auto normAreas = gp::lbo::voronoiAreas(mesh);
		mesh.pbind(refPos);
		A = normAreas.asDiagonal() * gp::eig::spIdentity(lbo.rows());
		msolver.compute(A);
		tsolver.compute(A + timeStep * lbo);
	}

	/// V x (sample size * source size)
	template <typename T = int>
	Eigen::MatrixXd compute(const std::vector<T>& source) {
		size_t n = (size_t)lbo.get().rows();
		size_t s = source.size();
		Eigen::MatrixXd whks(n, sampleSize * s);

		Eigen::SparseMatrix<double> S(n, source.size());
		std::vector<Eigen::Triplet<double>> triplets;
		for (size_t i = 0; i < source.size(); i++)
			triplets.emplace_back(static_cast<int>(source[i]), static_cast<int>(i), 1.0);
		S.setFromTriplets(triplets.begin(), triplets.end());

		whks.leftCols(s) = msolver.solve(lbo.get() * S);

		for (size_t i = 1; i < sampleSize; i++) {
			whks.block(0, i*s, n, s) = tsolver.solve(A*whks.block(0, (i-1)*s, n, s));
		}

		whks.array().rowwise() /= (A * whks.cwiseAbs()).colwise().sum().array();
		whks.array() /= (whks.maxCoeff() - whks.minCoeff());
		return whks;
	}

	/// V x sampleSize
	template <typename T = int>
	Eigen::MatrixXd computeFlat(const  std::vector<T>& source) {
		size_t n = (size_t)lbo.get().rows();
		size_t s = 1;
		Eigen::MatrixXd whks(n, sampleSize * s);

		Eigen::SparseMatrix<double> S(n, 1);
		std::vector<Eigen::Triplet<double>> triplets;
		for (size_t i = 0; i < source.size(); i++)
			triplets.emplace_back(static_cast<int>(source[i]), 0, 1.0 / source.size());
		S.setFromTriplets(triplets.begin(), triplets.end());

		whks.leftCols(s) = msolver.solve(lbo.get() * S);

		for (size_t i = 1; i < sampleSize; i++) {
			whks.block(0, i*s, n, s) = tsolver.solve(A*whks.block(0, (i-1)*s, n, s));
		}

		whks.array().rowwise() /= (A * whks.cwiseAbs()).colwise().sum().array();
		whks.array() /= (whks.maxCoeff() - whks.minCoeff());
		return whks;
	}

};
#endif
}