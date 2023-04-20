#pragma once
#include <thread>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/laplacian.hpp>
#include <mygp/nearest_neighbors.hpp>
#include <mygp/lbfgs.hpp>

namespace gp::fm {
// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXd computeFunctionalMap(
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs, 
	const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs, 
	const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, const double lambda=1.0) {
		
	int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();

	Eigen::MatrixXd fmap(k1, k2);
	Eigen::MatrixXd srcProj = srcEvecs.transpose() * srcAreas.asDiagonal() * srcDescs;
	Eigen::MatrixXd fixedA = srcProj * srcProj.transpose();

	Eigen::MatrixXd trgProj = trgEvecs.transpose() * trgAreas.asDiagonal() * trgDescs;

	Eigen::MatrixXd B = srcProj * trgProj.transpose();

	//#pragma omp parallel for
	for (int i = 0; i < k2; i++) {
		Eigen::VectorXd R = (srcEvals.array() - trgEvals(i)).square();
		Eigen::MatrixXd A = lambda * R.asDiagonal().toDenseMatrix() + fixedA;
		fmap.col(i) = A.lu().solve(B.col(i));
	}
	return fmap.transpose();
}

//GLP.fmap = gp::computeFunctionalMap(
//	trgData.evals.topRows(GLP.zoomStart), trgData.evecs.leftCols(GLP.zoomStart), trgData.areas, trgData.descs.rowwise().normalized(),
//	srcData.evals.topRows(GLP.zoomStart), srcData.evecs.leftCols(GLP.zoomStart), srcData.areas, srcData.descs.rowwise().normalized(),
//	GLP.commutOp);
// point-wise map S -> T, functional map T -> S
// fmapTS = gp::zoomout(fmapTS, trgData.evecs, srcData.evecs, srcData.areas, i, zoomStride);
inline Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap,
	const Eigen::MatrixXd& srcEvecs, 
	const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const double alpha = 1.0) {

	//auto corres = nn::nearestNeighbors(
	//	fmap.block(0, 0, current, current) * srcEvecs.leftCols(current).transpose(),
	//	trgEvecs.leftCols(current).transpose()
	//);
	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose()).cast<float>(),
		trgEvecs.leftCols(current).cast<float>()
	);
		
	return 
		alpha *
		gp::eig::pinvSolve(trgEvecs.leftCols(current+stride), srcEvecs.leftCols(current+stride)(corres, Eigen::all));
		//trgEvecs.leftCols(current+stride).transpose() *	trgAreas.asDiagonal() *(
		//srcEvecs.leftCols(current+stride)(corres, Eigen::all))
		;
}

/// point-wise map S -> T, functional map T -> S
/// 
/// @return [fmapST, fmapTS]
inline void zoomoutBijective(
	Eigen::MatrixXd& fmapST, Eigen::MatrixXd& fmapTS,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& srcAreas,
	const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const double alphaTS = 1.0, double alphaST = 0, 
	double factorTS = 0, double factorST = 0) {

	auto corresTS = gp::spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs.leftCols(current) * fmapST.block(0, 0, current, current).transpose()).cast<float>(),
		 trgEvecs.leftCols(current).cast<float>()
	);
	auto corresST = gp::spart::FaissKNN<false>::nearestNeighbors(
		(trgEvecs.leftCols(current) * fmapTS.block(0, 0, current, current).transpose()).cast<float>(),
		 srcEvecs.leftCols(current).cast<float>()
	);

	fmapST =
		gp::eig::pinvSolve(gp::eig::vConcatDenseMat(trgEvecs.leftCols(current), trgEvecs.leftCols(current)(corresST, Eigen::all)),
		eig::vConcatDenseMat(srcEvecs.leftCols(current)(corresTS, Eigen::all), srcEvecs.leftCols(current)));
		//gp::eig::hConcatDenseMat(trgEvecs.leftCols(current).transpose() * trgAreas.asDiagonal(),
		//(trgEvecs.leftCols(current).transpose() * trgAreas.asDiagonal())(Eigen::all, corresST)) *
		//eig::vConcatDenseMat(srcEvecs.leftCols(current)(corresTS, Eigen::all), srcEvecs.leftCols(current));

	fmapTS = 
		gp::eig::pinvSolve(gp::eig::vConcatDenseMat(srcEvecs.leftCols(current), srcEvecs.leftCols(current)(corresTS, Eigen::all)),
		eig::vConcatDenseMat(trgEvecs.leftCols(current)(corresST, Eigen::all), trgEvecs.leftCols(current)));
		//gp::eig::hConcatDenseMat(srcEvecs.leftCols(current).transpose() * srcAreas.asDiagonal(),
		//(srcEvecs.leftCols(current).transpose() * srcAreas.asDiagonal())(Eigen::all, corresTS))*
		//eig::vConcatDenseMat(trgEvecs.leftCols(current)(corresST, Eigen::all), trgEvecs.leftCols(current));

	if (factorST != 0) {
		fmapST.leftCols(1).fill(0.0);
		fmapST(0, 0) = factorST;
	}
	if (factorTS != 0) {
		fmapTS.leftCols(1).fill(0.0);
		fmapTS(0, 0) = factorTS;
	}

	corresTS = gp::spart::FaissKNN<false>::nearestNeighbors(
		eig::hConcatDenseMat(srcEvecs.leftCols(current) * fmapST.transpose(), srcEvecs.leftCols(current) * fmapTS).cast<float>(),
		eig::hConcatDenseMat(trgEvecs.leftCols(current), trgEvecs.leftCols(current)).cast<float>()
	);
	corresST = gp::spart::FaissKNN<false>::nearestNeighbors(
		eig::hConcatDenseMat(trgEvecs.leftCols(current) * fmapTS.transpose(), trgEvecs.leftCols(current) * fmapST).cast<float>(),
		eig::hConcatDenseMat(srcEvecs.leftCols(current), srcEvecs.leftCols(current)).cast<float>()
	);

	fmapST = gp::eig::pinvSolve(trgEvecs.leftCols(current+stride), srcEvecs.leftCols(current+stride)(corresTS, Eigen::all));
	fmapTS = gp::eig::pinvSolve(srcEvecs.leftCols(current+stride), trgEvecs.leftCols(current+stride)(corresST, Eigen::all));
}

/// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXd refineICP(
	const Eigen::MatrixXd& fmap,
	const Eigen::MatrixXd& srcEvecs,
	const Eigen::MatrixXd& trgEvecs,
	const Eigen::VectorXd& trgAreas) {

	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs * fmap.transpose()).cast<float>(),
		trgEvecs.cast<float>()
	);

	//trgEvecs.transpose() * trgAreas.asDiagonal() * srcEvecs(corres, Eigen::all);
	Eigen::MatrixXd S = gp::eig::pinvSolve(trgEvecs, srcEvecs(corres, Eigen::all));

	Eigen::JacobiSVD<Eigen::MatrixXd> solver(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto V = solver.matrixV();
	auto U = solver.matrixU();

	Eigen::VectorXd D = Eigen::VectorXd::Ones(S.rows());
	D(S.rows()-1) = (V * U.transpose()).determinant() > 0 ? 1.0 : -1.0;

	// (VDU^T)^T
	return (U * D.asDiagonal() * V.transpose());
}

inline Eigen::MatrixXd coordinateManifoldHarmonics(
	const Eigen::MatrixXd& pos, const Eigen::MatrixXd& evecs, const Eigen::VectorXd& areas) {
	size_t k = (size_t)pos.cols();
	Eigen::MatrixXd error =	pos - evecs * (evecs.transpose() * (areas.asDiagonal() * pos));
	Eigen::MatrixXd cmh = evecs;

	for (size_t i = 0; i < k; i++) {
		Eigen::VectorXd mcmh = error.col(i) - cmh * (cmh.transpose() * (areas.asDiagonal() * error.col(i)));

		mcmh = mcmh.array() / sqrt(mcmh.transpose()* (areas.asDiagonal()* mcmh));

		cmh = eig::hConcatDenseMat(cmh, mcmh);
	}

	return cmh;
}

inline double evecFactor(const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
	const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas) {
	return (((srcEvecs(0, 0)*trgEvecs(0, 0))>0) - ((srcEvecs(0, 0)*trgEvecs(0, 0))<0)) * sqrt(trgAreas.sum() / srcAreas.sum());
}

class DescCommutPreserveFmap : public gp::opt::MinConfPQN {
	int k1, k2;

	std::vector<Eigen::MatrixXd> srcBuffer, trgBuffer;
	Eigen::MatrixXd srcFeats, trgFeats;
	Eigen::VectorXd constFactor;
	Eigen::MatrixXd lambda;
	double coeffEvalCommut;
	double coeffDescCommut;
	double coeffDescPreserve;

	double op(const Eigen::VectorXd& x, Eigen::VectorXd& grad) override {
		double fx = 0.0;

		Eigen::MatrixXd fmap = x.reshaped(k2, k1);

		fx += coeffDescPreserve * 0.5 * (fmap * srcFeats - trgFeats).squaredNorm();
		Eigen::MatrixXd gradMat = coeffDescPreserve * (fmap * srcFeats - trgFeats)* srcFeats.transpose();

		for (size_t i = 0; i < srcBuffer.size(); i++) {
			Eigen::MatrixXd&& temp = trgBuffer[i] * fmap - fmap * srcBuffer[i];
			fx += coeffDescCommut * 0.5 * temp.squaredNorm();
			gradMat += coeffDescCommut * (trgBuffer[i].transpose() * temp - temp * srcBuffer[i].transpose());
		}

		fx += coeffEvalCommut * 0.5 * (fmap.array().square() * lambda.array()).sum();
		gradMat.array() += coeffEvalCommut * fmap.array()* lambda.array();

		grad = Eigen::Map<Eigen::VectorXd>(gradMat.data(), gradMat.size());
		return fx;
	}

	Eigen::VectorXd project(Eigen::VectorXd x) override {
		x.topRows(k2) = constFactor;
		return x;
	}

public:

	DescCommutPreserveFmap(
		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs,
		const int subsampleSize = std::numeric_limits<int>::max(),
		const double coeffEvalCommut = 1e-3, const double coeffDescCommut = 1.0, const double coeffDescPreserve = 1e-1) :
		k1((int)srcEvals.rows()), k2((int)trgEvals.rows()),
		coeffEvalCommut(coeffEvalCommut), coeffDescCommut(coeffDescCommut), coeffDescPreserve(coeffDescPreserve) {

		// ----------------------------------------
		constFactor = Eigen::VectorXd::Zero(k2);
		constFactor[0] = evecFactor(srcEvecs, srcAreas, trgEvecs, trgAreas);

		// ----------------------------------------
		int ds = std::min(subsampleSize <= 0 ? std::numeric_limits<int>::max() : subsampleSize, (int)srcDescs.cols());
		double step = (double)srcDescs.cols() / ds;
		double enume = 0.0;

		srcBuffer.resize(ds);
		trgBuffer.resize(ds);
		Eigen::MatrixXd&& sEtA = srcEvecs.transpose()* srcAreas.asDiagonal();
		Eigen::MatrixXd&& tEtA = trgEvecs.transpose()* trgAreas.asDiagonal();
		for (int i = 0; i < ds; i++, enume+=step) {
			int j = (int)ceil(enume);
			srcBuffer[i] = sEtA * srcDescs.col(j).asDiagonal() * srcEvecs;
			trgBuffer[i] = tEtA * trgDescs.col(j).asDiagonal() * trgEvecs;
		}

		srcFeats = sEtA * srcDescs;
		trgFeats = tEtA * trgDescs;

		// ----------------------------------------
		double maxEval = std::max(srcEvals.maxCoeff(), trgEvals.maxCoeff());
		Eigen::VectorXd srcLa = srcEvals.array() / maxEval;
		Eigen::VectorXd trgLa = trgEvals.array() / maxEval;
			
		// normal version
		//lambda = (trgLa.replicate(1, k1) - srcLa.transpose().replicate(k2, 1)).array().square();
			
		// complex resolvent mask version
		srcLa = srcLa.array().sqrt(); trgLa = trgLa.array().sqrt();
		lambda = ((trgLa.array() / (1.0 + trgLa.array().square())).replicate(1, k1) 
			-	(srcLa.array() / (1.0 + srcLa.array().square())).transpose().replicate(k2, 1)).array().square();
		lambda.array() += ((1.0 / (1.0 + trgLa.array().square())).replicate(1, k1)
			-	(1.0 / (1.0 + srcLa.array().square())).transpose().replicate(k2, 1)).array().square();

		lambda.normalized();

	}

	// WARNINGS : After this process, some resources will be cleared.
	Eigen::MatrixXd compute(const Eigen::MatrixXd& init, const size_t maxIter = 500) {
		Eigen::VectorXd x = init.reshaped(k1 * k2, 1);
		if (abs(x.sum()) <= 1e-10) x.reshaped(k2, k1).diagonal().fill(1.0);
		x[0] = this->constFactor[0];

		// relative weight 
		Eigen::MatrixXd fmap = x.reshaped(k2, k1);
		coeffDescPreserve /= 0.5 * (fmap * srcFeats - trgFeats).squaredNorm();
		coeffEvalCommut /= 0.5 * (fmap.array().square() *	lambda.array()).sum();
		double fx = 0.0;
		for (size_t i = 0; i < srcBuffer.size(); i++)
			fx += 0.5 * (trgBuffer[i] * fmap - fmap * srcBuffer[i]).squaredNorm();
		coeffDescCommut /= fx;

		// optimization
		this->minConfPQN(x, maxIter);

		// free resources
		srcBuffer.clear();
		srcBuffer.shrink_to_fit();
		trgBuffer.clear();
		trgBuffer.shrink_to_fit();
		srcFeats.resize(0, 0);
		trgFeats.resize(0, 0);
		lambda.resize(0, 0);
		constFactor.resize(0);
		this->freeResources();
		return x.reshaped(k2, k1);
	}
};

class DescCommutOrientPreserveFmap : public gp::opt::MinConfPQN {
	int k1, k2;

	std::vector<Eigen::MatrixXd> srcBuffer, trgBuffer;
	std::vector<Eigen::MatrixXd> srcOrient, trgOrient;
	Eigen::MatrixXd srcFeats, trgFeats;
	Eigen::MatrixXd lambda;
	Eigen::VectorXd constFactor;
	double coeffEvalCommut;
	double coeffDescCommut;
	double coeffDescPreserve;
	double coeffOrientPreserve;

	template <typename T>
	Eigen::SparseMatrix<double> orientationOperator(const heds::Mesh<T>& mesh,
		const Eigen::MatrixXi& tri, const Eigen::MatrixXd& gradMat,
		const Eigen::VectorXd& f) {
		//Eigen::MatrixXd grad = gp::grad::surfaceGradient(mesh, f).rowwise().normalized();
		//auto rgrad = gp::grad::rot90TanSpace(mesh, grad);

		size_t nv = mesh.getNumVerts();
		size_t nf = mesh.getNumFaces();

		Eigen::SparseMatrix<double> mat(nv, nv);
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(9 * nf);

		for (size_t i = 0; i < nf; i++) {
			auto v0 = tri(i, 0), v1 = tri(i, 1), v2 = tri(i, 2);
			Eigen::Vector3d gf = (gradMat.block<3, 3>(i*3, 0) * Eigen::Vector3d(f(v0), f(v1), f(v2)));

			double w0 = gf(0) / 6.0;
			double w1 = gf(1) / 6.0;
			double w2 = gf(2) / 6.0;

			triplets.emplace_back(v0, v0, (w0 + w1));
			triplets.emplace_back(v1, v1, (w1 + w2));
			triplets.emplace_back(v2, v2, (w2 + w0));
			triplets.emplace_back(v0, v1, -w1);
			triplets.emplace_back(v1, v0, -w1);
			triplets.emplace_back(v1, v2, -w2);
			triplets.emplace_back(v2, v1, -w2);
			triplets.emplace_back(v2, v0, -w0);
			triplets.emplace_back(v0, v2, -w0);
		}

		mat.setFromTriplets(triplets.begin(), triplets.end());

		return mat;
	}

	double op(const Eigen::VectorXd& x, Eigen::VectorXd& grad) override {
		double fx = 0.0;

		Eigen::MatrixXd fmap = x.reshaped(k2, k1);

		fx += coeffDescPreserve * 0.5 * (fmap * srcFeats - trgFeats).squaredNorm();
		Eigen::MatrixXd gradMat = coeffDescPreserve * (fmap * srcFeats - trgFeats)* srcFeats.transpose();

		for (size_t i = 0; i < srcBuffer.size(); i++) {
			Eigen::MatrixXd temp = trgBuffer[i] * fmap - fmap * srcBuffer[i];
			fx += coeffDescCommut * 0.5 * temp.squaredNorm();
			gradMat += coeffDescCommut * (trgBuffer[i].transpose() * temp - temp * srcBuffer[i].transpose());
			temp =  trgOrient[i] * fmap - fmap * srcOrient[i];
			fx += coeffOrientPreserve * 0.5 * temp.squaredNorm();
			gradMat += coeffOrientPreserve * (trgOrient[i].transpose() * temp - temp * srcOrient[i].transpose());
		}

		fx += coeffEvalCommut * 0.5 * (fmap.array().square() * lambda.array()).sum();
		gradMat.array() += coeffEvalCommut * fmap.array()* lambda.array();

		grad = Eigen::Map<Eigen::VectorXd>(gradMat.data(), gradMat.size());
		return fx;
	}

	Eigen::VectorXd project(Eigen::VectorXd x) override {
		x.topRows(k2) = constFactor;
		return x;
	}

public:
	DescCommutOrientPreserveFmap(
		const heds::Mesh<Eigen::MatrixX3d>& srcMesh, const heds::Mesh<Eigen::MatrixX3d>& trgMesh,
		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs,
		const int subsampleSize = std::numeric_limits<int>::max(),
		const double coeffEvalCommut = 1e-3, const double coeffDescCommut = 1.0, 
		const double coeffDescPreserve = 1e-1, const double coeffOrientPreserve = 1.0) :
		k1((int)srcEvals.rows()), k2((int)trgEvals.rows()),
		coeffEvalCommut(coeffEvalCommut), coeffDescCommut(coeffDescCommut), 
		coeffDescPreserve(coeffDescPreserve),	coeffOrientPreserve(coeffOrientPreserve) {

		// ----------------------------------------
		constFactor = Eigen::VectorXd::Zero(k2);
		constFactor[0] = evecFactor(srcEvecs, srcAreas, trgEvecs, trgAreas);

		// ----------------------------------------
		int ds = std::min(subsampleSize <= 0 ? std::numeric_limits<int>::max() : subsampleSize, (int)srcDescs.cols());
		double step = (double)srcDescs.cols() / ds;
		double enume = 0.0;

		srcBuffer.resize(ds, Eigen::MatrixXd(k1, k1));
		trgBuffer.resize(ds, Eigen::MatrixXd(k2, k2));
		srcOrient.resize(ds, Eigen::MatrixXd(k1, k1));
		trgOrient.resize(ds, Eigen::MatrixXd(k2, k2));
		Eigen::MatrixXi srcTri = srcMesh.getFaceIndexMat(), trgTri = trgMesh.getFaceIndexMat();
		Eigen::MatrixXd srcGrad = gp::grad::surfaceGradientOp(srcMesh), trgGrad = gp::grad::surfaceGradientOp(trgMesh);
		Eigen::MatrixXd&& sEtA = srcEvecs.transpose()* srcAreas.asDiagonal();
		Eigen::MatrixXd&& tEtA = trgEvecs.transpose()* trgAreas.asDiagonal();
		for (int i = 0; i < ds; i++, enume+=step) {
			int j = (int)ceil(enume);
			srcBuffer[i] = sEtA * srcDescs.col(j).asDiagonal() * srcEvecs;
			trgBuffer[i] = tEtA * trgDescs.col(j).asDiagonal() * trgEvecs;
			// original : EtA * A.inv() * orientOp * SrcEvecs;
			srcOrient[i] = srcEvecs.transpose() * orientationOperator(srcMesh, srcTri, srcGrad, srcDescs.col(j)) * srcEvecs;
			trgOrient[i] = trgEvecs.transpose() * orientationOperator(trgMesh, trgTri, trgGrad, trgDescs.col(j)) * trgEvecs;
		}

		srcFeats = sEtA * srcDescs;
		trgFeats = tEtA * trgDescs;

		// ----------------------------------------
		double maxEval = std::max(srcEvals.maxCoeff(), trgEvals.maxCoeff());
		Eigen::VectorXd srcLa = srcEvals.array() / maxEval;
		Eigen::VectorXd trgLa = trgEvals.array() / maxEval;

		// normal version
		//lambda = (trgLa.replicate(1, k1) - srcLa.transpose().replicate(k2, 1)).array().square();

		// complex resolvent mask version
		srcLa = srcLa.array().sqrt(); trgLa = trgLa.array().sqrt();
		lambda = ((trgLa.array() / (1.0 + trgLa.array().square())).replicate(1, k1)
			-	(srcLa.array() / (1.0 + srcLa.array().square())).transpose().replicate(k2, 1)).array().square();
		lambda.array() += ((1.0 / (1.0 + trgLa.array().square())).replicate(1, k1)
			-	(1.0 / (1.0 + srcLa.array().square())).transpose().replicate(k2, 1)).array().square();

		lambda.normalized();

	}

	// WARNINGS : After this process, some resources will be cleared.
	Eigen::MatrixXd compute(const Eigen::MatrixXd& init, const size_t maxIter = 500) {
		Eigen::VectorXd x = init.reshaped(k1 * k2, 1);
		if (abs(x.sum()) <= 1e-10) x.reshaped(k2, k1).diagonal().fill(1.0);
		x[0] = this->constFactor[0];

		// relative weight 
		Eigen::MatrixXd fmap = x.reshaped(k2, k1);
		coeffDescPreserve /= 0.5 * (fmap * srcFeats - trgFeats).squaredNorm();
		coeffEvalCommut /= 0.5 * (fmap.array().square() *	lambda.array()).sum();
		double fx1 = 0.0, fx2 = 0.0;
		for (size_t i = 0; i < srcBuffer.size(); i++) {
			fx1 += 0.5 * (trgBuffer[i] * fmap - fmap * srcBuffer[i]).squaredNorm();
			fx2 += 0.5 * (trgOrient[i] * fmap - fmap * srcOrient[i]).squaredNorm();
		}
		coeffDescCommut /= fx1;
		coeffOrientPreserve /= fx2;

		this->minConfPQN(x, maxIter);

		// free resources
		srcBuffer.clear();
		srcBuffer.shrink_to_fit();
		trgBuffer.clear();
		trgBuffer.shrink_to_fit();
		srcOrient.clear();
		srcOrient.shrink_to_fit();
		trgOrient.clear();
		trgOrient.shrink_to_fit();
		srcFeats.resize(0, 0);
		trgFeats.resize(0, 0);
		lambda.resize(0, 0);
		constFactor.resize(0);
		this->freeResources();
		return x.reshaped(k2, k1);
	}
};

}

namespace gp::fm::dopt {

// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXd conformalSolver(const Eigen::MatrixXd& fmap,
	const Eigen::VectorXd& srcEvals,
	const Eigen::MatrixXd& srcEvecs,
	const Eigen::VectorXd& trgEvals,
	const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const size_t iter = 5, const double alpha = 1.0) {
	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose()).cast<float>(),
		trgEvecs.leftCols(current).cast<float>()
	);

	size_t cs = current + stride;
	Eigen::VectorXd sevals = srcEvals.topRows(cs), tevals = trgEvals.topRows(cs);
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();

	Eigen::MatrixXd ret = fmap;
	for (size_t i = 0; i < iter; i++) {
		ret = alpha *
			gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all));
			//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *	
			//srcEvecs.leftCols(cs)(corres, Eigen::all)
			;

		corres = spart::FaissKNN<false>::nearestNeighbors(
			(srcEvecs.leftCols(cs) * sevals.asDiagonal() * ret.block(0, 0, cs, cs).transpose()).cast<float>(),
			(trgEvecs.leftCols(cs) * tevals.asDiagonal()).cast<float>()
		);
	}

	ret = alpha * 
		gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all))
		//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *
		//srcEvecs.leftCols(cs)(corres, Eigen::all)
		;

	return ret;
}

// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXcd conformalSolverCmplx(const Eigen::MatrixXcd& fmap,
	const Eigen::VectorXcd& srcEvals,
	const Eigen::MatrixXcd& srcEvecs,
	const Eigen::VectorXcd& trgEvals,
	const Eigen::MatrixXcd& trgEvecs, 
	const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const size_t iter = 5, const double alpha = 1.0) {

	Eigen::MatrixXcd tmp1 = srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).adjoint();
	Eigen::MatrixXcd tmp2 = trgEvecs.leftCols(current);

	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(tmp1.real(), tmp1.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(tmp2.real(), tmp2.imag()).cast<float>()
	);

	size_t cs = current + stride;
	Eigen::VectorXd sevals = srcEvals.topRows(cs).real(), tevals = trgEvals.topRows(cs).real();
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();

	Eigen::MatrixXcd ret = fmap;
	for (size_t i = 0; i < iter; i++) {
		ret = alpha * trgEvecs.leftCols(cs).adjoint() *
			trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all);

		tmp1 = srcEvecs.leftCols(cs) * sevals.asDiagonal() * ret.block(0, 0, cs, cs).adjoint();
		tmp2 = trgEvecs.leftCols(cs) * tevals.asDiagonal();
		corres = spart::FaissKNN<false>::nearestNeighbors(
			gp::eig::hConcatDenseMat(tmp1.real(), tmp1.imag()).cast<float>(),
			gp::eig::hConcatDenseMat(tmp2.real(), tmp2.imag()).cast<float>()
		);
	}

	ret = alpha * trgEvecs.leftCols(cs).adjoint() *
		trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all);

	return ret;
}

// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXd lapCommutSolver(const Eigen::MatrixXd& fmap,
	const Eigen::VectorXd& srcEvals,
	const Eigen::MatrixXd& srcEvecs,
	const Eigen::VectorXd& trgEvals,
	const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const size_t iter = 5, const double alpha = 1.0, 
	const bool lhs = true) {

	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose()).cast<float>(),
		trgEvecs.leftCols(current).cast<float>()
	);

	size_t cs = current + stride;
	Eigen::VectorXd sevals = srcEvals.topRows(cs), tevals = trgEvals.topRows(cs);
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();
	
	Eigen::MatrixXd ret = fmap;
	for (size_t i = 0; i < iter; i++) {
		ret = alpha *
			gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all));
		//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all)
		;

		if (lhs) {
			corres = spart::FaissKNN<false>::nearestNeighbors(
				(srcEvecs.leftCols(cs) * sevals.asDiagonal() * ret.block(0, 0, cs, cs).transpose()).cast<float>(),
				(trgEvecs.leftCols(cs) * tevals.asDiagonal()).cast<float>()
			);
		} else {
			corres = spart::FaissKNN<false>::nearestNeighbors(
				(srcEvecs.leftCols(cs) * sevals.asDiagonal()).cast<float>(),
				(trgEvecs.leftCols(cs) * tevals.asDiagonal() * ret.block(0, 0, cs, cs)).cast<float>()
			);
		}

		ret = alpha *
			gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all));
			//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all)
		;

		corres = spart::FaissKNN<false>::nearestNeighbors(
			(srcEvecs.leftCols(cs) * ret.block(0, 0, cs, cs).transpose()).cast<float>(),
			trgEvecs.leftCols(cs).cast<float>()
		);
	}

	ret = alpha *
		gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all));
	//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all)
	;

	return ret;
}

// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXd descLapOrthoSolver(const Eigen::MatrixXd& fmap,
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
	const Eigen::MatrixXd& srcDescs, const Eigen::VectorXd& srcAreas,
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
	const Eigen::MatrixXd& trgDescs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, 
	const size_t iter = 5, const double alpha = 1.0, 
	const double coeffDescs = 0.1, const double coeffLapcommut = 0.1) {
	
	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose()).cast<float>(),
		trgEvecs.leftCols(current).cast<float>()
	);

	size_t cs = current + stride;
	Eigen::VectorXd sevals = srcEvals.topRows(cs), tevals = trgEvals.topRows(cs);
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();

	Eigen::MatrixXd srcFeats = gp::eig::pinvSolve(srcEvecs.leftCols(cs), srcDescs);
	Eigen::MatrixXd trgFeats = gp::eig::pinvSolve(trgEvecs.leftCols(cs), trgDescs);

	Eigen::MatrixXd ret = fmap;

	for (size_t i = 0; i < iter; i++) {
		ret = alpha *
			gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all));
		//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all)
		;
		corres = spart::FaissKNN<false>::nearestNeighbors(
			gp::eig::hConcatDenseMat(
			coeffDescs * srcEvecs.leftCols(cs) * srcFeats,
			coeffLapcommut * srcEvecs.leftCols(cs) * sevals.asDiagonal() * ret.block(0, 0, cs, cs).transpose(),
			(srcEvecs.leftCols(cs) * ret.block(0, 0, cs, cs).transpose())).cast<float>(),
			gp::eig::hConcatDenseMat(
			coeffDescs * trgEvecs.leftCols(cs) * trgFeats,
			coeffLapcommut * trgEvecs.leftCols(cs) * tevals.asDiagonal(),
			trgEvecs.leftCols(cs)).cast<float>()
		);
	}

	ret = alpha *
		gp::eig::pinvSolve(trgEvecs.leftCols(cs), srcEvecs.leftCols(cs)(corres, Eigen::all));
	//trgEvecs.leftCols(cs).transpose() * trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all)
	;

	return ret;
}


// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXd descLapOrthoOrientSolver(const Eigen::MatrixXd& fmap,
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
	const Eigen::MatrixXd& srcDescs, const Eigen::VectorXd& srcAreas,
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
	const Eigen::MatrixXd& trgDescs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride,
	const double coeffOrient = 1.0,
	const double coeffDescs = 0.1, const double coeffLapcommut = 0.1,
	const size_t iter = 5, const double alpha = 1.0) {

	auto corres = spart::FaissKNN<false>::nearestNeighbors(
		(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose()).cast<float>(),
		trgEvecs.leftCols(current).cast<float>()
	);

	size_t cs = current + stride;
	Eigen::VectorXd sevals = srcEvals.topRows(cs), tevals = trgEvals.topRows(cs);
	sevals.array() /= sevals.sum();
	tevals.array() /= tevals.sum();

	Eigen::MatrixXd srcFeats = srcEvecs.leftCols(cs).transpose()* srcAreas.asDiagonal()* srcDescs;
	Eigen::MatrixXd trgFeats = trgEvecs.leftCols(cs).transpose()* trgAreas.asDiagonal()* trgDescs;

	Eigen::MatrixXd srcTmp = srcEvecs.leftCols(cs).array().colwise() * srcDescs.col(0).array();
	Eigen::MatrixXd trgTmp = trgEvecs.leftCols(cs).array().colwise() * trgDescs.col(0).array();

	Eigen::MatrixXd srcOp = srcEvecs.leftCols(cs).transpose()* srcAreas.asDiagonal() * srcTmp;
	Eigen::MatrixXd trgOp = trgEvecs.leftCols(cs).transpose()* trgAreas.asDiagonal() * trgTmp;

	Eigen::MatrixXd ret = fmap;

	for (size_t i = 0; i < iter; i++) {
		ret = alpha * trgEvecs.leftCols(cs).transpose() *
			trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all);

		corres = spart::FaissKNN<false>::nearestNeighbors(
			gp::eig::hConcatDenseMat(
			coeffDescs * srcEvecs.leftCols(cs) * srcFeats,
			coeffLapcommut * srcEvecs.leftCols(cs) * sevals.asDiagonal() * ret.block(0, 0, cs, cs).transpose(),
			coeffOrient * srcEvecs.leftCols(cs) * srcOp * ret.block(0, 0, cs, cs).transpose(),
			(srcEvecs.leftCols(cs) * ret.block(0, 0, cs, cs).transpose())).cast<float>(),
			gp::eig::hConcatDenseMat(
			coeffDescs * trgEvecs.leftCols(cs) * trgFeats,
			coeffLapcommut * trgEvecs.leftCols(cs) * tevals.asDiagonal(),
			coeffOrient * trgEvecs.leftCols(cs) * trgOp,
			trgEvecs.leftCols(cs)).cast<float>()
		);
	}

	ret = alpha * trgEvecs.leftCols(cs).transpose() *
		trgAreas.asDiagonal() *	srcEvecs.leftCols(cs)(corres, Eigen::all);

	return ret;
}

}



// complex version
namespace gp::fm {

inline std::vector<faiss::Index::idx_t> cmap2pmap(const Eigen::MatrixXcd& cmap, const Eigen::MatrixXcd& srcCevecs, const Eigen::MatrixXcd& trgCevecs, const size_t current) {
	Eigen::MatrixXcd tmp1 = srcCevecs.leftCols(current) * cmap.block(0, 0, current, current).adjoint();
	Eigen::MatrixXcd tmp2 = trgCevecs.leftCols(current);
	return gp::spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(tmp1.real(), tmp1.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(tmp2.real(), tmp2.imag()).cast<float>());

}

// Eigen::SparseMatrix<Eigen::dcomplex> srcGradOp = gp::grad::vertexGradientOp(srcData.mesh);
// Eigen::MatrixXd invPhi = srcData.evecs.transpose() * srcData.areas.asDiagonal();
// std::vector<Eigen::MatrixXcd> srcFeats(numeigs, Eigen::MatrixXcd(numeigs, numeigs));
// srcFeats[i] = invPhi * (srcGradOp * srcData.evecs.col(i)).asDiagonal() * srcData.cevecs.conjugate();
//
// convert real functional map to complex map.
// D = pinv(evecs) * dot_as_2d(grad(f), cevecs)
// solve |CstDt - DsCst|^2
inline Eigen::MatrixXcd fmap2cmap(const Eigen::MatrixXd& fmap,
	const std::vector<Eigen::MatrixXcd>& srcFeats, const std::vector<Eigen::MatrixXcd>& trgFeats) {
	size_t k1 = (size_t)fmap.rows(), k2 = (size_t)fmap.cols();

	Eigen::MatrixXcd rhs(k1 * k1, k2);
	Eigen::MatrixXcd lhs(k1 * k1, k1);

	for (size_t i = 0; i < k1; i++) {
		lhs.block(i*k1, 0, k1, k2) = fmap * srcFeats[i].block(0, 0, k2, k2);

		Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(k1, k2);
		for (size_t j = 0; j < k2; j++) mat += trgFeats[j].block(0, 0, k1, k1) * fmap(j, i);
		rhs.block(i*k1, 0, k1, k2) = mat;
	}

	Eigen::BDCSVD<Eigen::MatrixXcd> solver(lhs.adjoint() * rhs, Eigen::ComputeThinU | Eigen::ComputeThinV);
	return (solver.matrixV() * solver.matrixU().adjoint()).conjugate();
}

inline Eigen::MatrixXd cmap2fmap(const Eigen::MatrixXcd& cmap,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXcd& srcCevecs, 
	const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXcd& trgCevecs) {
	Eigen::MatrixXcd tmp1 = srcCevecs.leftCols(cmap.cols()) * cmap.adjoint();
	Eigen::MatrixXcd tmp2 = trgCevecs.leftCols(cmap.rows());
	auto corres = gp::spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(tmp1.real(), tmp1.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(tmp2.real(), tmp2.imag()).cast<float>());

	return gp::eig::pinvSolve(trgEvecs.leftCols(cmap.rows()), srcEvecs.leftCols(cmap.cols())(corres, Eigen::all));
}

// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXcd computeFunctionalMapCmplx(
	const Eigen::VectorXcd& srcEvals, const Eigen::MatrixXcd& srcEvecs,
	const Eigen::VectorXd& srcAreas,  const Eigen::MatrixXcd& srcDescs,
	const Eigen::VectorXcd& trgEvals, const Eigen::MatrixXcd& trgEvecs,
	const Eigen::VectorXd& trgAreas,  const Eigen::MatrixXcd& trgDescs, const double lambda=1.0) {
	int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();

	Eigen::MatrixXcd fmap(k1, k2);
	Eigen::MatrixXcd srcProj = srcEvecs.adjoint() * srcAreas.asDiagonal() * srcDescs;
	Eigen::MatrixXcd fixedA = srcProj * srcProj.adjoint();

	Eigen::MatrixXcd trgProj = trgEvecs.adjoint() * trgAreas.asDiagonal() * trgDescs;

	Eigen::MatrixXcd B = srcProj * trgProj.adjoint();

	for (int i = 0; i < k2; i++) {
		Eigen::VectorXcd R = (srcEvals.array() - trgEvals(i)).abs2();
		Eigen::MatrixXcd A = lambda * R.asDiagonal().toDenseMatrix() + fixedA;
		fmap.col(i) = A.lu().solve(B.col(i));
	}

	return fmap.adjoint();
}

// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXcd zoomoutCmplx(
	const Eigen::MatrixXcd& fmap, const Eigen::MatrixXcd& srcEvecs,
	const Eigen::MatrixXcd& trgEvecs, const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const double alpha = 1.0) {

	Eigen::MatrixXcd tmp1 = srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).adjoint();
	Eigen::MatrixXcd tmp2 = trgEvecs.leftCols(current);
	auto corres = gp::spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(tmp1.real(), tmp1.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(tmp2.real(), tmp2.imag()).cast<float>());

	//// low performance
	//auto corres = gp::nn::nearestNeighborsCmplx(
	//	fmap.block(0, 0, current, current) * srcEvecs.leftCols(current).adjoint(),
	//	trgEvecs.leftCols(current).adjoint()
	//);

	return alpha *
		gp::eig::pinvSolve(trgEvecs.leftCols(current+stride), srcEvecs.leftCols(current+stride)(corres, Eigen::all));
		//trgEvecs.leftCols(current+stride).adjoint() *	trgAreas.asDiagonal() *
		//srcEvecs.leftCols(current+stride)(corres, Eigen::all);
}

/// point-wise map S -> T, functional map T -> S
inline Eigen::MatrixXcd refineICPCmplx(
	const Eigen::MatrixXcd& fmap,
	const Eigen::MatrixXcd& srcEvecs,
	const Eigen::MatrixXcd& trgEvecs,
	const Eigen::VectorXd& trgAreas) {

	Eigen::MatrixXcd tmp = srcEvecs * fmap.adjoint();
	auto corres = gp::spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(tmp.real(), tmp.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(trgEvecs.real(), trgEvecs.imag()).cast<float>());

	Eigen::MatrixXcd S = trgEvecs.adjoint() * trgAreas.asDiagonal() * srcEvecs(corres, Eigen::all);

	Eigen::BDCSVD<Eigen::MatrixXcd> solver(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto V = solver.matrixV();
	auto U = solver.matrixU();

	//Eigen::VectorXd D = Eigen::VectorXd::Ones(S.rows());
	//D(S.rows()-1) = (V * U.adjoint()).determinant().real() > 0 ? 1.0 : -1.0;

	// (VDU^T)^T
	return (U * V.adjoint());
}

/// point-wise map S -> T, functional map T -> S
inline void refineICPQmap(
	Eigen::MatrixXd& fmap,
	Eigen::MatrixXcd& cmap,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXcd& srcCevecs, const std::vector<Eigen::MatrixXcd>& srcFeats,
	const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXcd& trgCevecs, const std::vector<Eigen::MatrixXcd>& trgFeats,
	const Eigen::VectorXd& trgAreas) {

	cmap = fmap2cmap(fmap, srcFeats, trgFeats);
	Eigen::MatrixXcd tmp1 = srcCevecs * cmap.adjoint();
	Eigen::MatrixXcd tmp2 = trgCevecs;
	auto corres = gp::spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(tmp1.real(), tmp1.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(tmp2.real(), tmp2.imag()).cast<float>());

	Eigen::MatrixXd S = gp::eig::pinvSolve(trgEvecs, srcEvecs(corres, Eigen::all));

	Eigen::JacobiSVD<Eigen::MatrixXd> solver(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto V = solver.matrixV();
	auto U = solver.matrixU();

	Eigen::VectorXd D = Eigen::VectorXd::Ones(S.rows());
	D(S.rows()-1) = (V * U.transpose()).determinant() > 0 ? 1.0 : -1.0;

	fmap = (U * D.asDiagonal() * V.transpose());
}

// point-wise map S -> T, functional map T -> S
inline void zoomoutQmap(
	Eigen::MatrixXd& fmap, Eigen::MatrixXcd& cmap,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXcd& srcCevecs, const std::vector<Eigen::MatrixXcd>& srcFeats, 
	const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXcd& trgCevecs, const std::vector<Eigen::MatrixXcd>& trgFeats, 
	const Eigen::VectorXd& trgAreas,
	const size_t current, const size_t stride, const double alpha = 1.0) {

	cmap = fmap2cmap(fmap, srcFeats, trgFeats);
	Eigen::MatrixXcd tmp1 = srcCevecs.leftCols(current) * cmap.block(0, 0, current, current).adjoint();
	Eigen::MatrixXcd tmp2 = trgCevecs.leftCols(current);
	auto corres = gp::spart::FaissKNN<false>::nearestNeighbors(
		gp::eig::hConcatDenseMat(//srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose(), 
		tmp1.real(), tmp1.imag()).cast<float>(),
		gp::eig::hConcatDenseMat(//trgEvecs.leftCols(current), 
		tmp2.real(), tmp2.imag()).cast<float>());

	fmap = alpha * gp::eig::pinvSolve(trgEvecs.leftCols(current+stride), srcEvecs.leftCols(current+stride)(corres, Eigen::all));
	//cmap = alpha *
	//	trgCevecs.leftCols(current+stride).adjoint() *
	//	trgAreas.asDiagonal() *
	//	srcCevecs.leftCols(current+stride)(corres, Eigen::all);
}


/// point-wise map S -> T, functional map T -> S
/// @return [fmapST, fmapTS]
inline void zoomoutBijectiveQmap(
	Eigen::MatrixXd& fmapST, Eigen::MatrixXd& fmapTS,
	Eigen::MatrixXcd& cmapST, Eigen::MatrixXcd& cmapTS,
	const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXcd& srcCevecs, const std::vector<Eigen::MatrixXcd>& srcFeats, const Eigen::MatrixXd& srcAreas,
	const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXcd& trgCevecs, const std::vector<Eigen::MatrixXcd>& trgFeats, const Eigen::MatrixXd& trgAreas,
	const size_t current, const size_t stride) {
	cmapST = fmap2cmap(fmapST, srcFeats, trgFeats);
	cmapTS = fmap2cmap(fmapTS, trgFeats, srcFeats);

	auto corresTS = cmap2pmap(cmapST, srcCevecs, trgCevecs, current);
	auto corresST = cmap2pmap(cmapTS, trgCevecs, srcCevecs, current);

	fmapST =
		gp::eig::pinvSolve(gp::eig::vConcatDenseMat(trgEvecs.leftCols(current), trgEvecs.leftCols(current)(corresST, Eigen::all)),
		eig::vConcatDenseMat(srcEvecs.leftCols(current)(corresTS, Eigen::all), srcEvecs.leftCols(current)));
	//gp::eig::hConcatDenseMat(trgEvecs.leftCols(current).transpose() * trgAreas.asDiagonal(),
	//(trgEvecs.leftCols(current).transpose() * trgAreas.asDiagonal())(Eigen::all, corresST)) *
	//eig::vConcatDenseMat(srcEvecs.leftCols(current)(corresTS, Eigen::all), srcEvecs.leftCols(current));

	fmapTS =
		gp::eig::pinvSolve(gp::eig::vConcatDenseMat(srcEvecs.leftCols(current), srcEvecs.leftCols(current)(corresTS, Eigen::all)),
		eig::vConcatDenseMat(trgEvecs.leftCols(current)(corresST, Eigen::all), trgEvecs.leftCols(current)));
	//gp::eig::hConcatDenseMat(srcEvecs.leftCols(current).transpose() * srcAreas.asDiagonal(),
	//(srcEvecs.leftCols(current).transpose() * srcAreas.asDiagonal())(Eigen::all, corresTS))*
	//eig::vConcatDenseMat(trgEvecs.leftCols(current)(corresST, Eigen::all), trgEvecs.leftCols(current));

	corresTS = gp::spart::FaissKNN<false>::nearestNeighbors(
		eig::hConcatDenseMat(srcEvecs.leftCols(current) * fmapST.transpose(), srcEvecs.leftCols(current) * fmapTS).cast<float>(),
		eig::hConcatDenseMat(trgEvecs.leftCols(current), trgEvecs.leftCols(current)).cast<float>()
	);
	corresST = gp::spart::FaissKNN<false>::nearestNeighbors(
		eig::hConcatDenseMat(trgEvecs.leftCols(current) * fmapTS.transpose(), trgEvecs.leftCols(current) * fmapST).cast<float>(),
		eig::hConcatDenseMat(srcEvecs.leftCols(current), srcEvecs.leftCols(current)).cast<float>()
	);

	fmapST = gp::eig::pinvSolve(trgEvecs.leftCols(current+stride), srcEvecs.leftCols(current+stride)(corresTS, Eigen::all));

	fmapTS = gp::eig::pinvSolve(srcEvecs.leftCols(current+stride), trgEvecs.leftCols(current+stride)(corresST, Eigen::all));
}

// point-wise map S -> T, functional map T -> S
class DescCommutPreserveFmapCmplx : public gp::opt::MinConfPQNCmplx {
	int k1, k2;

	std::vector<Eigen::MatrixXcd> srcBuffer, trgBuffer;
	Eigen::MatrixXcd srcFeats, trgFeats;
	Eigen::MatrixXd lambda;
	double coeffEvalCommut;
	double coeffDescCommut;
	double coeffDescPreserve;

	double op(const Eigen::VectorXcd& x, Eigen::VectorXcd& grad) override {
		double fx = 0.0;

		Eigen::MatrixXcd fmap = x.reshaped(k2, k1);

		fx += coeffDescPreserve * 0.5 * (fmap * srcFeats - trgFeats).squaredNorm();
		Eigen::MatrixXcd gradMat = coeffDescPreserve * (fmap * srcFeats - trgFeats)* srcFeats.adjoint();

		for (size_t i = 0; i < srcBuffer.size(); i++) {
			Eigen::MatrixXcd&& temp = trgBuffer[i] * fmap - fmap * srcBuffer[i];
			fx += coeffDescCommut * 0.5 * temp.squaredNorm();
			gradMat += coeffDescCommut * (trgBuffer[i].adjoint() * temp - temp * srcBuffer[i].adjoint());
		}

		fx += coeffEvalCommut * 0.5 * (fmap.array().abs2() * lambda.array()).sum();
		gradMat.array() += coeffEvalCommut * fmap.array()* lambda.array();

		grad = Eigen::Map<Eigen::VectorXcd>(gradMat.data(), gradMat.size());
		return fx;
	}

	Eigen::VectorXcd project(Eigen::VectorXcd x) override {
		return x;
	}

public:

	DescCommutPreserveFmapCmplx(
		const Eigen::VectorXcd& srcEvals, const Eigen::MatrixXcd& srcEvecs, const Eigen::VectorXcd& srcAreas, const Eigen::MatrixXcd& srcDescs,
		const Eigen::VectorXcd& trgEvals, const Eigen::MatrixXcd& trgEvecs, const Eigen::VectorXcd& trgAreas, const Eigen::MatrixXcd& trgDescs,
		const int subsampleSize = std::numeric_limits<int>::max(),
		const double coeffEvalCommut = 1e-3, const double coeffDescCommut = 1.0, const double coeffDescPreserve = 1e-1) :
		k1((int)srcEvals.rows()), k2((int)trgEvals.rows()),
		coeffEvalCommut(coeffEvalCommut), coeffDescCommut(coeffDescCommut), coeffDescPreserve(coeffDescPreserve) {

		// ----------------------------------------
		int ds = std::min(subsampleSize <= 0 ? std::numeric_limits<int>::max() : subsampleSize, (int)srcDescs.cols());
		double step = (double)srcDescs.cols() / ds;
		double enume = 0.0;

		srcBuffer.resize(ds);
		trgBuffer.resize(ds);
		Eigen::MatrixXcd&& sEtA = srcEvecs.adjoint()* srcAreas.asDiagonal();
		Eigen::MatrixXcd&& tEtA = trgEvecs.adjoint()* trgAreas.asDiagonal();
		for (int i = 0; i < ds; i++, enume+=step) {
			int j = (int)ceil(enume);
			srcBuffer[i] = sEtA * srcDescs.col(j).asDiagonal() * srcEvecs;
			trgBuffer[i] = tEtA * trgDescs.col(j).asDiagonal() * trgEvecs;
		}
		
		srcFeats = sEtA * srcDescs;
		trgFeats = tEtA * trgDescs;
		
		//// ----------------------------------------
		double maxEval = std::max(srcEvals.real().maxCoeff(), trgEvals.real().maxCoeff());
		Eigen::VectorXd srcLa = srcEvals.real().array() / maxEval;
		Eigen::VectorXd trgLa = trgEvals.real().array() / maxEval;
		
		// normal version
		lambda = (trgLa.replicate(1, k1) - srcLa.transpose().replicate(k2, 1)).array().square();

		// complex resolvent mask version
		//srcLa = srcLa.array().sqrt(); trgLa = trgLa.array().sqrt();
		//lambda = ((trgLa.array() / (1.0 + trgLa.array().square())).replicate(1, k1)
		//	-	(srcLa.array() / (1.0 + srcLa.array().square())).transpose().replicate(k2, 1)).array().square();
		//lambda.array() += ((1.0 / (1.0 + trgLa.array().square())).replicate(1, k1)
		//	-	(1.0 / (1.0 + srcLa.array().square())).transpose().replicate(k2, 1)).array().square();

		lambda.normalized();

	}

	// WARNINGS : After this process, some resources will be cleared.
	Eigen::MatrixXcd compute(const Eigen::MatrixXcd& init, const size_t maxIter = 500) {
		Eigen::VectorXcd x = init.reshaped(k1 * k2, 1);

		// relative weight 
		Eigen::MatrixXcd fmap = x.reshaped(k2, k1);
		coeffDescPreserve /= 0.5 * (fmap * srcFeats - trgFeats).squaredNorm();
		coeffEvalCommut /= 0.5 * (fmap.array().abs2() *	lambda.array()).sum();
		double fx = 0.0;
		for (size_t i = 0; i < srcBuffer.size(); i++)
			fx += 0.5 * (trgBuffer[i] * fmap - fmap * srcBuffer[i]).squaredNorm();
		coeffDescCommut /= fx;

		// optimization
		this->minConfPQN(x, maxIter);

		// free resources
		srcBuffer.clear();
		srcBuffer.shrink_to_fit();
		trgBuffer.clear();
		trgBuffer.shrink_to_fit();
		srcFeats.resize(0, 0);
		trgFeats.resize(0, 0);
		lambda.resize(0, 0);
		this->freeResources();
		return x.reshaped(k2, k1);

	}
};

}

namespace gp::experimental {

//	// point-wise map S -> T, functional map T -> S
//	Eigen::MatrixXd computeFunctionalMap(
//		const Eigen::SparseMatrix<double>& srcGop, const Eigen::SparseMatrix<double>& trgGop,
//		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
//		const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
//		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
//		const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, const double lambda=1.0) {
//
//		int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();
//
//		Eigen::MatrixXd fmap(k1, k2);
//		Eigen::MatrixXd srcProj = srcEvecs.transpose() * srcAreas.asDiagonal() * srcDescs;
//		Eigen::MatrixXd fixedA = srcProj * srcProj.transpose();
//
//		Eigen::MatrixXd trgProj = trgEvecs.transpose() * trgAreas.asDiagonal() * trgDescs;
//
//		Eigen::MatrixXd B = srcProj * trgProj.transpose();
//
//		#pragma omp parallel for
//		for (int i = 0; i < k2; i++) {
//			Eigen::VectorXd R = (srcEvals.array() - trgEvals(i)).square();
//			Eigen::MatrixXd A = lambda * R.asDiagonal().toDenseMatrix() + fixedA;
//			fmap.col(i) = A.lu().solve(B.col(i));
//		}
//		return fmap.transpose();
//	}
//
//	/*
//	// point-wise map S -> T, functional map T -> S
//	Eigen::MatrixXd computeFunctionalMap(
//		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
//		const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs, const Eigen::MatrixXd& srcGMH,
//		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
//		const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, const Eigen::MatrixXd& trgGMH, 
//		const double lambda=1.0) {
//
//		int k1 = (int)trgEvals.rows(), k2 = (int)srcEvals.rows();
//
//		Eigen::MatrixXd fmap(k1, k2);
//		Eigen::MatrixXd trgProj = trgDescs.transpose() * trgAreas.asDiagonal() * trgEvecs;
//		Eigen::MatrixXd srcProj = srcDescs.transpose() * srcAreas.asDiagonal() * srcEvecs;
//
//		Eigen::MatrixXd fixedA = trgProj.transpose() * trgProj;
//		Eigen::MatrixXd B = trgProj.transpose() * srcProj;
//
//		#pragma omp parallel for
//		for (int i = 0; i < k2; i++) {
//			Eigen::VectorXd R = Eigen::pow(trgEvals.array() - srcEvals(i), 2);
//			Eigen::MatrixXd A = lambda * R.asDiagonal().toDenseMatrix() + fixedA;
//			fmap.col(i) = A.ldlt().solve(B.col(i));
//		}
//		return fmap;
//	}
//
//	// point-wise map S -> T, functional map T -> S
//	Eigen::MatrixXd computeFunctionalMap(
//		const Eigen::MatrixXd& oldmap,
//		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
//		const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
//		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
//		const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, const double lambda=1.0, const double mu = 0.001) {
//
//		int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();
//
//		Eigen::MatrixXd fmap(k1, k2);
//		Eigen::MatrixXd srcProj = srcEvecs.transpose() * srcAreas.asDiagonal() * srcDescs;
//		Eigen::MatrixXd fixedA = srcProj * srcProj.transpose();
//
//		Eigen::MatrixXd trgProj = trgEvecs.transpose() * trgAreas.asDiagonal() * trgDescs;
//		Eigen::MatrixXd B = srcProj * trgProj.transpose() + mu * oldmap.transpose();
//
//		#pragma omp parallel for
//		for (int i = 0; i < k2; i++) {
//			Eigen::VectorXd R = Eigen::pow(srcEvals.array() - trgEvals(i), 2);
//			Eigen::MatrixXd A = lambda * R.asDiagonal().toDenseMatrix() + fixedA;
//			fmap.col(i) = A.ldlt().solve(B.col(i));
//		}
//		return fmap.transpose();
//	}
//
//	// point-wise map S -> T, functional map T -> S
//	Eigen::MatrixXd computeFunctionalMapCommut(
//		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs,
//		const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
//		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs,
//		const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, 
//		const double gamma = 1e-3, const double beta = 1e-1, const double alpha = 1.0) {
//
//		int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();
//
//		Eigen::MatrixXd fmap(k1, k2);
//		Eigen::MatrixXd srcProj = srcEvecs.transpose() * srcAreas.asDiagonal() * srcDescs;
//		Eigen::MatrixXd fixedA = srcProj * srcProj.transpose();
//
//		Eigen::MatrixXd trgProj = trgEvecs.transpose() * trgAreas.asDiagonal() * trgDescs;
//		Eigen::MatrixXd B = srcProj * trgProj.transpose();
//		
//		int cd = (int)srcDescs.cols();
//		Eigen::MatrixXd commutMat = Eigen::MatrixXd::Zero(k2 * k1, k2 * k1);
//
//		for (int i = 0; i < cd; i++) {
//			Eigen::MatrixXd X = (srcEvecs.transpose() * srcAreas.asDiagonal() * (srcDescs.col(i).asDiagonal() * srcEvecs)).transpose();
//			Eigen::MatrixXd Y = (trgEvecs.transpose() * trgAreas.asDiagonal() * (trgDescs.col(i).asDiagonal() * trgEvecs));
//			
//			Eigen::MatrixXd temp1 = Eigen::MatrixXd::Zero(k2 * k1, k2 * k1);
//			Eigen::MatrixXd temp2 = Eigen::MatrixXd::Zero(k2 * k1, k2 * k1);
//			
//			for (int j = 0; j < k2; j++) {
//				temp1.block(j * k1, j * k1, k1, k1) = X;
//			}
//			
//			for (int r = 0; r < k2; r++) {
//				for (int j = 0; j < k2; j++) {
//					temp2.block(j * k1, r * k1, k1, k1) = Y(j, r) * Eigen::VectorXd::Ones(k1).asDiagonal();
//				}
//			}
//			
//			commutMat.array() += Eigen::pow(temp1.array() - temp2.array(), 2);
//		}
//
//		Eigen::MatrixXd lA = alpha*commutMat.transpose();
//		for (int i = 0; i < k2; i++) {
//			Eigen::VectorXd R = Eigen::pow(srcEvals.array() - trgEvals(i), 2);
//			lA.block(i*k1, i*k1, k1, k1) += gamma * R.asDiagonal().toDenseMatrix() + fixedA;
//		}
//		
//		Eigen::VectorXd b = B.reshaped(k1 * k2, 1);
//		Eigen::VectorXd f = lA.lu().solve(b);
//		fmap = f.reshaped(k2, k1);
//
//		return fmap.transpose();
//	}
//	*/
//
//	/// point-wise map S -> T, functional map T -> S
//	/// PMapS2T = refineV2VMap(fmapT2S, trgEvecs, srcEvecs);
//	std::vector<int> refineV2VMap(const Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& trgEvecs) {
//		return nn::nearestNeighbors(fmap * srcEvecs.transpose(), trgEvecs.transpose());
//	}
//
//	std::vector<int> refineV2VMapBijective(const Eigen::MatrixXd& fmapST, const Eigen::MatrixXd& fmapTS,
//		const Eigen::MatrixXd& srcEvecs, const Eigen::MatrixXd& trgEvecs) {
//		return nn::nearestNeighbors(
//			gp::eig::hConcatDenseMat(srcEvecs * fmapST.transpose(), srcEvecs * fmapTS).transpose(),
//			gp::eig::hConcatDenseMat(trgEvecs, trgEvecs).transpose()
//		);
//	}
//

//	Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap,
//		const std::vector<std::vector<uint32_t>>& srcNeighbs,
//		const std::vector<std::vector<uint32_t>>& trgNeighbs,
//		const Eigen::MatrixXd& srcEvecs,
//		const Eigen::MatrixXd& trgEvecs, const Eigen::VectorXd& trgAreas,
//		const size_t current, const size_t stride, const double alpha = 1.0) {
//
//		auto corres = nn::nearestNeighbors(
//			srcNeighbs, trgNeighbs,
//			fmap.block(0, 0, current, current) * srcEvecs.leftCols(current).transpose(),
//			trgEvecs.leftCols(current).transpose()
//		);
//		//auto corres = spart::FaissKNN<false>::nearestNeighbors(
//		//	(srcEvecs.leftCols(current) * fmap.block(0, 0, current, current).transpose()).cast<float>(),
//		//	trgEvecs.leftCols(current).cast<float>()
//		//);
//
//		return alpha *
//			trgEvecs.leftCols(current+stride).transpose() *
//			trgAreas.asDiagonal() *
//			srcEvecs.leftCols(current+stride)(corres, Eigen::all);
//	}

//	/*
//	//GLP.fmap = gp::computeFunctionalMap(
//	//	trgData.evals.topRows(GLP.zoomStart), trgData.evecs.leftCols(GLP.zoomStart), trgData.areas, trgData.descs.rowwise().normalized(),
//	//	srcData.evals.topRows(GLP.zoomStart), srcData.evecs.leftCols(GLP.zoomStart), srcData.areas, srcData.descs.rowwise().normalized(),
//	//	GLP.commutOp);
//	// point-wise map S -> T, functional map T -> S
//	// fmapTS = gp::zoomout(fmapTS, srcData.evecs, srcData.areas, trgData.evecs, i, zoomStride);
//	Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs,
//		const Eigen::VectorXd& srcAreas,
//		const Eigen::MatrixXd& trgEvecs,
//		const Eigen::VectorXd& trgAreas,
//		const size_t current, const size_t stride, 
//		const Eigen::Vector<uint32_t, -1> srcSegments,
//		const Eigen::Vector<uint32_t, -1> trgSegments,
//		const double alpha = 1.0) {
//		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();
//
//		auto corres = nn::nearestNeighbors(
//			fmap.block(0, 0, current, current) * trgEvecs.leftCols(current).transpose(),
//			srcEvecs.leftCols(current).transpose()
//			, trgSegments, srcSegments
//		);
//
//		auto P = eig::extractMatrix((int)ns, (int)nt, corres, 1);
//
//		return alpha *
//			srcEvecs.leftCols(current+stride).transpose() *
//			srcAreas.asDiagonal() *
//			(P * (
//			trgEvecs.leftCols(current+stride)));
//	}
//
//	// point-wise map S -> T, functional map T -> S
//	// fmapTS = gp::zoomout(fmapTS, srcData.evecs, srcData.areas, trgData.evecs, i, zoomStride);
//	Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
//		const Eigen::MatrixXd& trgEvecs, const size_t current, const size_t stride, 
//		const Eigen::Vector<uint32_t, -1> srcSegments,
//		const Eigen::Vector<uint32_t, -1> trgSegments,
//		const double alpha = 1.0) {
//		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();
//
//		auto corres = nn::nearestNeighbors(
//			(trgEvecs.block(0, 0, nt, current) * fmap.block(0, 0, current, current)).transpose(),
//			srcEvecs.block(0, 0, ns, current).transpose()
//			,trgSegments, 
//			srcSegments
//		);
//
//		auto P = eig::extractMatrix((int)nt, (int)ns, corres, 0);
//
//		return alpha * trgEvecs.block(0, 0, nt, current+stride).transpose() *
//			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));
//
//	}
//
//	// point-wise map S -> T, functional map T -> S
//// fmapTS = gp::zoomout(fmapTS, srcData.evecs, srcData.areas, trgData.evecs, i, zoomStride);
//	Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcPos,
//		const Eigen::MatrixXd& trgEvecs, const Eigen::MatrixXd& trgPos, const size_t current, const size_t stride,
//		const Eigen::Vector<uint32_t, -1> srcSegments,
//		const Eigen::Vector<uint32_t, -1> trgSegments,
//		const double alpha = 1.0) {
//		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();
//
//		auto corres = nn::nearestNeighbors(
//			(trgEvecs.block(0, 0, nt, current) * fmap.block(0, 0, current, current)).transpose(),
//			srcEvecs.block(0, 0, ns, current).transpose(),
//			trgPos.transpose(), srcPos.transpose(),
//			trgSegments,
//			srcSegments,
//			0.01
//		);
//
//		auto P = eig::extractMatrix((int)nt, (int)ns, corres, 0);
//
//		return alpha * trgEvecs.block(0, 0, nt, current+stride).transpose() *
//			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));
//
//	}
//	*/

//	/*
//	////GLP.fmap = gp::computeFunctionalMap(
//	////	srcData.evals.topRows(GLP.zoomStart), srcData.evecs.leftCols(GLP.zoomStart), srcData.areas, srcData.descs.rowwise().normalized(),
//	////	trgData.evals.topRows(GLP.zoomStart), trgData.evecs.leftCols(GLP.zoomStart), trgData.areas, trgData.descs.rowwise().normalized(),
//	////	GLP.commutOp);
//	Eigen::MatrixXd refineICP(
//		const Eigen::MatrixXd& fmap,
//		const Eigen::MatrixXd& srcEvecs,
//		const Eigen::VectorXd& srcAreas, 
//		const Eigen::MatrixXd& trgEvecs,
//		const Eigen::Vector<uint32_t, -1> srcSegments,
//		const Eigen::Vector<uint32_t, -1> trgSegments
//	) {
//		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();
//
//		Eigen::MatrixXd X = fmap * srcEvecs.transpose();
//		auto corres = nn::nearestNeighbors(
//			X,
//			trgEvecs.transpose(), 
//			srcSegments, trgSegments
//		);
//
//		auto P = eig::extractMatrix((int)ns, (int)nt, corres, 0);
//
//		Eigen::MatrixXd S = srcEvecs.transpose() * srcAreas.asDiagonal() * P * trgEvecs;
//
//		Eigen::JacobiSVD<Eigen::MatrixXd> solver(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
//		auto V = solver.matrixV();
//		auto U = solver.matrixU();
//
//		Eigen::VectorXd D = Eigen::VectorXd::Ones(S.rows());
//		D(S.rows()-1) = (V * U.transpose()).determinant();
//
//		return (V * D.asDiagonal() * U.transpose());
//	}
//
//	// point-wise map S -> T, functional map T -> S
//	// fmapTS = gp::zoomout(fmapTS, srcData.evecs, srcData.areas, trgData.evecs, i, zoomStride);
//	void zoomoutMT(Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
//		const Eigen::MatrixXd& trgEvecs, const size_t current, const size_t stride, 
//		std::vector<int>& corres, bool& signal, const double ratio = 1.0) {
//		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();
//
//		Eigen::SparseMatrix<double> P(nt, ns);
//		P.reserve(ns);
//		std::vector<Eigen::Triplet<double>> ptriplets;
//		ptriplets.reserve(ns);
//
//		corres = nn::nearestNeighbors(
//			(trgEvecs.block(0, 0, nt, current) * fmap).transpose(),
//			srcEvecs.block(0, 0, ns, current).transpose()
//		);
//
//		for (int j = 0; j < (int)ns; j++) ptriplets.emplace_back(corres[j], j, 1.0);
//		P.setFromTriplets(ptriplets.begin(), ptriplets.end());
//
//		fmap = ratio * trgEvecs.block(0, 0, nt, current+stride).transpose() *
//			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));
//
//		signal = true;
//	}
//
//	// point-wise map S -> T, functional map T -> S
//	// fmapTS = gp::zoomout(fmapTS, srcData.evecs, srcData.areas, trgData.evecs, i, zoomStride);
//	void zoomoutMTS(Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
//		const Eigen::MatrixXd& trgEvecs, const size_t current, const size_t stride,
//		std::vector<int>& corres, bool& signal, 
//		const Eigen::Vector<uint32_t, -1> srcSegments, 
//		const Eigen::Vector<uint32_t, -1> trgSegments,
//		const double ratio = 1.0) {
//		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();
//
//		Eigen::SparseMatrix<double> P(nt, ns);
//		P.reserve(ns);
//		std::vector<Eigen::Triplet<double>> ptriplets;
//		ptriplets.reserve(ns);
//
//		corres = nn::nearestNeighbors(
//			(trgEvecs.block(0, 0, nt, current) * fmap).transpose(),
//			srcEvecs.block(0, 0, ns, current).transpose()
//			,trgSegments, srcSegments
//		);
//
//		for (int j = 0; j < (int)ns; j++) ptriplets.emplace_back(corres[j], j, 1.0);
//		P.setFromTriplets(ptriplets.begin(), ptriplets.end());
//
//		fmap = ratio * trgEvecs.block(0, 0, nt, current+stride).transpose() *
//			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));
//
//		signal = true;
//	}

//	Eigen::MatrixXd coordManHarmonics(
//		const Eigen::MatrixXd& pos, const Eigen::MatrixXd& evecs, const Eigen::VectorXd& areas, const size_t block) {
//		size_t k = (size_t)pos.cols();
//		Eigen::MatrixXd error =	pos - evecs.leftCols(block) * (evecs.leftCols(block).transpose() * (areas.asDiagonal() * pos));
//		Eigen::MatrixXd cmh = evecs;
//
//		for (size_t i = 0; i < k; i++) {
//			Eigen::VectorXd mcmh = error.col(i) - cmh * (cmh.transpose() * (areas.asDiagonal() * error.col(i)));
//
//			mcmh = mcmh.array()/sqrt(mcmh.transpose()* (areas.asDiagonal()* mcmh));
//
//			cmh = eig::hConcatDenseMat(cmh, mcmh);
//		}
//
//		return cmh.rightCols(k);
//	}

}


