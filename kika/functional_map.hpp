#pragma once

#include <kika/eigutil.hpp>
#include <kika/surface_mesh.hpp>
#include <kika/diff_geometry.hpp>
#include <kika/faiss_wrapper.hpp>
#include <kika/nanoflann_wrapper.hpp>
 
/// generate functional map
namespace kika::fm {

// compute functional maps
// min ||F_trg * C - F_src ||^2 + lambda * ||L_trg.T * C - (C * L_src).T||^2
// @return fmap : (trg -> src) (k_trg x k_src)
//
// (Ref) Functional Maps: A Flexible Representation of Maps Between Shapes [Ovsjanikov, M. et al. 2012]
inline Eigen::MatrixXd functionalMap(
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& srcDescs,
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXd& trgDescs,
	const double lambda=1.0) {

	int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();

	Eigen::MatrixXd fmap(k2, k1);
	Eigen::MatrixXd trgProj = kika::eig::pinvSolve(trgEvecs, trgDescs);
	Eigen::MatrixXd fixedA = trgProj * trgProj.transpose();

	Eigen::MatrixXd srcProj = kika::eig::pinvSolve(srcEvecs, srcDescs);
	Eigen::MatrixXd B = trgProj * srcProj.transpose();

	#pragma omp parallel for
	for (int i = 0; i < k1; i++) {
		Eigen::VectorXd R = (trgEvals.array() - srcEvals(i)).square();
		Eigen::MatrixXd A = fixedA;
		A.diagonal().array() += lambda * R.array();
		fmap.col(i) = A.lu().solve(B.col(i));
	}
	return fmap;
}

}

/// recover point-wise map
namespace kika::fm {

// recover point-wise map by nearest neighbors
// @param srcData : srcEvecs * fmap.transpose(); (functionla map (trg -> src))
// @param pmap : point-wise map (src -> trg) (n_trg x 1)
inline std::vector<uint32_t> nearestNeighbors(
	const Eigen::MatrixXd& srcData, const Eigen::MatrixXd& trgData) {
	return knn::FaissKNN<false>::nearestNeighbors<uint32_t>(srcData, trgData);
}

// recover point-wise map by fast sinkhorn filter
// @param srcData : srcEvecs * fmap.transpose() (functionla map (trg -> src))
// @param pmap : point-wise map (src -> trg) (n_trg x 1)
// (Ref) Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence With Functional Maps [Pai, G, et al. 2021]
inline std::vector<uint32_t> fastSinkhorn(
	const Eigen::MatrixXd& srcData,
	const Eigen::MatrixXd& trgData,
	const size_t k = 40, const size_t iter = 40, const double lambda = 400) {

	Eigen::SparseMatrix<double> kernel(srcData.rows(), trgData.rows());
	std::vector<Eigen::Triplet<double>> triplets;

	auto setKernelTriplets = [&k, &lambda](const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
		std::vector<Eigen::Triplet<double>>& triplets, const bool transpose = false) {

		std::vector<std::vector<uint32_t>> knnIndices;
		std::vector<std::vector<float>> knnDistances;

		knn::FaissKNN<false> tree(Y);
		tree.getKNearestNeighbors(X, k, knnIndices, knnDistances);

		double l = lambda;
		std::vector<double> vals(X.rows());
		#pragma omp parallel for
		for (int i = 0; i < (int)vals.size(); i++) {
			vals[i] = *std::max_element(knnDistances[i].begin(), knnDistances[i].end());
		}
		l /= *std::max_element(vals.begin(), vals.end());

		for (Eigen::Index i = 0; i < X.rows(); i++) {
			for (size_t j = 0; j < knnIndices[i].size(); j++) {
				if (transpose)
					triplets.emplace_back((int)knnIndices[i][j], (int)i, 0.5 * std::exp(-l * knnDistances[i][j]));
				else
					triplets.emplace_back((int)i, (int)knnIndices[i][j], 0.5 * std::exp(-l * knnDistances[i][j]));
			}
		}
	};

	setKernelTriplets(srcData, trgData, triplets);
	setKernelTriplets(trgData, srcData, triplets, true);
	kernel.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::VectorXd alpha = Eigen::VectorXd::Constant(trgData.rows(), 1. / trgData.rows());
	Eigen::VectorXd beta = Eigen::VectorXd::Constant(srcData.rows(), 1. / srcData.rows());
	Eigen::VectorXd  u = alpha, v = beta;
	for (size_t i = 0; i < iter; i++) {
		u = (alpha.array() / (kernel.transpose() * v).array()).eval();
		v = (beta.array() / (kernel * u).array()).eval();
	}

	Eigen::SparseMatrix<double> P = (v.asDiagonal()* kernel* u.asDiagonal());
	return kika::eig::spArgMax(P, false);
}

// recover point-wise map by sinkhorn filter(full compute kernel)
// @param srcData : srcEvecs * fmap.transpose() (functionla map (trg -> src))
// @param pmap : point-wise map (src -> trg) (n_trg x 1)
inline std::vector<uint32_t> fullSinkhorn(
	const Eigen::MatrixXd& srcData,
	const Eigen::MatrixXd& trgData,
	const size_t k = 0, const size_t iter = 40, const double lambda = 400) {
	Eigen::MatrixXd kernel(srcData.rows(), trgData.rows());

	for (Eigen::Index i = 0; i < kernel.cols(); i++) {
		kernel.col(i) = (srcData.rowwise() - trgData.row(i)).rowwise().squaredNorm();
	}

	double l = lambda / kernel.maxCoeff();
	kernel = Eigen::exp(-l * kernel.array());

	Eigen::VectorXd alpha = Eigen::VectorXd::Constant(trgData.rows(), 1. / trgData.rows());
	Eigen::VectorXd beta = Eigen::VectorXd::Constant(srcData.rows(), 1. / srcData.rows());
	Eigen::VectorXd  u = alpha, v = beta;
	for (size_t i = 0; i < iter; i++) {
		u = (alpha.array() / (kernel.transpose() * v).array()).eval();
		v = (beta.array() / (kernel * u).array()).eval();
		
	}

	Eigen::MatrixXd P = v.asDiagonal() * kernel * u.asDiagonal();
	std::vector<uint32_t> corres(trgData.rows());
	for (size_t i = 0; i < corres.size(); i++) P.col(i).maxCoeff(&corres[i]);
	return corres;
}

struct NearestNeighbors {
	std::vector<uint32_t> operator()(const Eigen::MatrixXd& srcData, const Eigen::MatrixXd& trgData) const {
		return nearestNeighbors(srcData, trgData);
	}
};

struct FastSinkhorn {
	const size_t k = 40;
	const size_t iter = 40;
	const double lambda = 400.;
	std::vector<uint32_t> operator()(const Eigen::MatrixXd& srcData, const Eigen::MatrixXd& trgData) const {
		return fastSinkhorn(srcData, trgData, k, iter, lambda);
	}
};

struct FullSinkhorn {
	const size_t iter = 40;
	const double lambda = 400.;
	std::vector<uint32_t> operator()(const Eigen::MatrixXd& srcData, const Eigen::MatrixXd& trgData) const {
		return fullSinkhorn(srcData, trgData, 0, iter, lambda);
	}
};

}

/// refine functional map
namespace kika::fm {

// refine functional map by zoomout
// @param fmap : functionla map (trg -> src) (k_trg x k_src)
template <class T = NearestNeighbors>
Eigen::MatrixXd icp(
	const Eigen::MatrixXd& fmap,
	const Eigen::MatrixXd& srcEvecs,
	const Eigen::MatrixXd& trgEvecs, const T& p2pFunc = T{}) {
	auto corres = p2pFunc(srcEvecs * fmap.transpose(), trgEvecs);
	Eigen::MatrixXd S = eig::pinvSolve(trgEvecs, srcEvecs(corres, Eigen::all));

	Eigen::JacobiSVD<Eigen::MatrixXd> solver(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto V = solver.matrixV();
	auto U = solver.matrixU();

	Eigen::VectorXd D = Eigen::VectorXd::Ones(S.rows());
	D(S.rows()-1) = (V * U.transpose()).determinant() > 0 ? 1.0 : -1.0;

	return (U * D.asDiagonal() * V.transpose());
}

// refine functional map by zoomout
// @param fmap : functionla map (trg -> src) (k_trg x k_src)
// @return fmap : (k_trg + stride x k_src + stride)
// (Ref) ZoomOut: Spectral Upsampling for Efficient Shape Correspondence[Melzi, S, et al. 2019]
template <class T = NearestNeighbors>
Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& trgEvecs,
	const size_t current, const size_t stride = 1, const T& p2pFunc = T{}) {
	
	auto corres = p2pFunc(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose(), trgEvecs.leftCols(current));

	return kika::eig::pinvSolve(trgEvecs.leftCols(current+stride), 
		srcEvecs.leftCols(current+stride)(corres, Eigen::all));
}

}

// generate functional map by discrete optimization (examples)
// 
// (Ref) Discrete Optimization for Shape Matching[Ren, J. 2021]
namespace kika::fm {

// minimize area preserving energy (orthogonality)
// min E_ortho = ||C * C.T - I||^2
// @param corres : point-wise map (src -> trg) (trg x 1)
// @return fmap :  functional map (trg -> src) (k_trg x k_src)
template <typename F = NearestNeighbors>
inline Eigen::MatrixXd discreteOptimizeOrtho(
	std::vector<uint32_t>& corres,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& trgEvecs,
	const size_t current, const size_t stride, const size_t iter = 5, F p2pFunc = F{}) {

	size_t r = current + stride;
	
	Eigen::MatrixXd fmap;
	for (size_t i = 0; i < iter+1; i++) {
		fmap = kika::eig::pinvSolve(trgEvecs.leftCols(r), srcEvecs.leftCols(r)(corres, Eigen::all));
		corres = p2pFunc(srcEvecs.leftCols(r) * fmap.transpose(), trgEvecs.leftCols(r));
	}


	return fmap;
}

// minimize angle preserving energy (conformality)
// min E_conformal = ||C * L_src * C.T - L_trg||^2
// @param corres : point-wise map (src -> trg) (trg x 1)
// @return fmap :  functional map (trg -> src) (k_trg x k_src)
template <typename F = NearestNeighbors>
inline Eigen::MatrixXd discreteOptimizeConformal(
	std::vector<uint32_t>& corres,
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
	const size_t current, const size_t stride, const size_t iter = 5, F p2pFunc = F{}) {

	size_t r = current + stride;

	Eigen::VectorXd sevals = srcEvals.topRows(r), tevals = trgEvals.topRows(r);
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();

	Eigen::MatrixXd fmap;
	for (size_t i = 0; i < iter+1; i++) {
		fmap = kika::eig::pinvSolve(trgEvecs.leftCols(r), srcEvecs.leftCols(r)(corres, Eigen::all));
		corres = p2pFunc(srcEvecs.leftCols(r) * sevals.asDiagonal() * fmap.transpose(), 
			trgEvecs.leftCols(r) * tevals.asDiagonal());
	}

	return fmap;
}


// minimize descriptors preserving energy
// min E_desc = ||C * C.T - I||^2 + alpha * ||F_trg * C - F_src||^2
// @param corres : point-wise map (src -> trg) (trg x 1)
// @return fmap :  functional map (trg -> src) (k_trg x k_src)
template <typename F = NearestNeighbors>
inline Eigen::MatrixXd discreteOptimizeDescs(
	std::vector<uint32_t>& corres,
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& srcDescs, 
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXd& trgDescs, 
	const size_t current, const size_t stride, const size_t iter = 5, const double alpha = 1e-3, F p2pFunc = F{}) {

	size_t r = current + stride;

	Eigen::VectorXd sevals = srcEvals.topRows(r), tevals = trgEvals.topRows(r);
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();

	Eigen::MatrixXd srcData(srcEvecs.rows(), srcDescs.cols() + r);
	Eigen::MatrixXd trgData(trgEvecs.rows(), trgDescs.cols() + r);
	srcData.leftCols(srcDescs.cols()) = alpha * eig::pinvSolve(srcEvecs, srcDescs);
	trgData.leftCols(trgDescs.cols()) = alpha * eig::pinvSolve(trgEvecs, trgDescs);
	trgData.rightCols(r) = trgEvecs.leftCols(r);

	Eigen::MatrixXd fmap;
	for (size_t i = 0; i < iter+1; i++) {
		fmap = kika::eig::pinvSolve(trgEvecs.leftCols(r), srcEvecs.leftCols(r)(corres, Eigen::all));
		srcData.rightCols(r) = srcEvecs.leftCols(r) * fmap.transpose();
		corres = p2pFunc(srcData, trgData);
	}

	return fmap;
}

}

/// generate manifold harmonics
namespace kika::fm {

// generate coordinate manifold harmonics
// @return cmh : |V|x(k+3) [evecs, cmh];
//
// (Ref) CMH: Coordinates Manifold Harmonics for Functional Remeshing[Riccardo, M. et al. 2019]
inline Eigen::MatrixXd coordinateManifoldHarmonics(const Eigen::MatrixX3d& pos, const Eigen::MatrixXd& evecs) {
	size_t k = (size_t)pos.cols();
	Eigen::MatrixX3d error =	pos - evecs * eig::pinvSolve(evecs, pos);
	Eigen::MatrixXd cmh = evecs;

	for (size_t i = 0; i < k; i++) {
		Eigen::VectorXd mcmh = error.col(i) - cmh * eig::pinvSolve(cmh, error.col(i));
		eig::hAppendDenseMat(cmh, mcmh.normalized());
	}

	return cmh;
}
}
