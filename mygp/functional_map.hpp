#pragma once
#include <fstream>
#include <filesystem>
#include <string>
#include <thread>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <mygp/nearest_neighbors.hpp>

namespace gp {
	Eigen::MatrixXd computeFunctionalMap(
		const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcEvecs, 
		const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
		const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgEvecs, 
		const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, const double lambda=1.0) {
		
		int k1 = (int)srcEvals.rows(), k2 = (int)trgEvals.rows();

		Eigen::MatrixXd fmap(k1, k2);
		Eigen::MatrixXd srcProj = srcEvecs.transpose() * srcAreas.asDiagonal() * srcDescs;
		Eigen::MatrixXd trgProj = trgEvecs.transpose() * trgAreas.asDiagonal() * trgDescs;
		
		Eigen::MatrixXd fixedA = srcProj * srcProj.transpose();
		Eigen::MatrixXd B = srcProj * trgProj.transpose();

		#pragma omp parallel for
		for (int i = 0; i < k2; i++) {
			Eigen::VectorXd R = Eigen::pow(srcEvals.array() - trgEvals(i), 2);
			Eigen::MatrixXd A = lambda * R.asDiagonal().toDenseMatrix() + fixedA;
			fmap.col(i) = A.ldlt().solve(B.col(i));
		}

		return fmap;
	}

	Eigen::MatrixXd computeFunctionalMap(
		const Eigen::MatrixXd& srcEvecs,
		const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
		const Eigen::MatrixXd& trgEvecs,
		const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs, const double lambda=1.0) {

		int k1 = (int)srcEvecs.cols(), k2 = (int)trgEvecs.cols();

		Eigen::MatrixXd fmap(k1, k2);
		Eigen::MatrixXd srcProj = srcEvecs.transpose() * srcAreas.asDiagonal() * srcDescs;
		Eigen::MatrixXd trgProj = trgEvecs.transpose() * trgAreas.asDiagonal() * trgDescs;

		Eigen::MatrixXd fixedA = srcProj * srcProj.transpose();
		Eigen::MatrixXd B = srcProj * trgProj.transpose();

		//fmap = fixedA.ldlt().solve(B);

		#pragma omp parallel for
		for (int i = 0; i < k2; i++) {
			Eigen::MatrixXd A = fixedA;
			fmap.col(i) = A.ldlt().solve(B.col(i));
		}

		return fmap;
	}

	// fmapTS = gp::zoomout(fmapTS, srcData.evecs, srcData.areas, trgData.evecs, i, zoomStride);
	Eigen::MatrixXd zoomout(const Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
													const Eigen::MatrixXd& trgEvecs, const size_t current, const size_t stride) {
		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();

		Eigen::SparseMatrix<double> P(nt, ns);
		P.reserve(ns);
		std::vector<Eigen::Triplet<double>> ptriplets;
		ptriplets.reserve(ns);

		auto corres = nearestNeighbors(
			(trgEvecs.block(0, 0, nt, current) * fmap).transpose(),
			 srcEvecs.block(0, 0, ns, current).transpose()
		);

		//ptriplets.clear();
		//P.resize(nt, ns);
		for (int j = 0; j < (int)ns; j++) ptriplets.emplace_back(corres[j], j, 1.0);
		P.setFromTriplets(ptriplets.begin(), ptriplets.end());
		
		return trgEvecs.block(0, 0, nt, current+stride).transpose() *
			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));

		//fmapTS.resize(i+1, i+1);
		//fmapTS.fill(0.0);
		//for (size_t j = 0; j < segmentCounter.size(); j++) {
		//	Eigen::VectorXd srcMask = Eigen::VectorXd::Zero(src.n);
		//	Eigen::VectorXd trgMask = Eigen::VectorXd::Zero(trg.n);
		//	for (size_t t = 0; t < src.n; t++) srcMask(t) = (srcData.segments[t] == j);
		//	for (size_t t = 0; t < trg.n; t++) trgMask(t) = (trgData.segments[t] == j);
		//	
		//	fmapTS += (trgMask.asDiagonal() * trgData.evecs.block(0, 0, trg.n, i+zoomStride)).transpose() *
		//		(P * (srcData.areas.asDiagonal() * (srcMask.asDiagonal() * srcData.evecs.block(0, 0, src.n, i+zoomStride))));
		//}

	}


	void zoomoutMT(Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
		const Eigen::MatrixXd& trgEvecs, const size_t current, const size_t stride, 
		std::vector<int>& corres, bool& signal, const double ratio = 1.0) {
		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();

		Eigen::SparseMatrix<double> P(nt, ns);
		P.reserve(ns);
		std::vector<Eigen::Triplet<double>> ptriplets;
		ptriplets.reserve(ns);

		corres = nearestNeighbors(
			(trgEvecs.block(0, 0, nt, current) * fmap).transpose(),
			srcEvecs.block(0, 0, ns, current).transpose()
		);

		for (int j = 0; j < (int)ns; j++) ptriplets.emplace_back(corres[j], j, 1.0);
		P.setFromTriplets(ptriplets.begin(), ptriplets.end());

		fmap = ratio * trgEvecs.block(0, 0, nt, current+stride).transpose() *
			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));

		signal = true;
	}

	void zoomoutMTS(Eigen::MatrixXd& fmap, const Eigen::MatrixXd& srcEvecs, const Eigen::VectorXd& srcAreas,
		const Eigen::MatrixXd& trgEvecs, const size_t current, const size_t stride,
		std::vector<int>& corres, bool& signal, 
		const Eigen::Vector<uint32_t, -1> srcSegments, 
		const Eigen::Vector<uint32_t, -1> trgSegments,
		const double ratio = 1.0) {
		auto ns = srcEvecs.rows(), nt = trgEvecs.rows();

		Eigen::SparseMatrix<double> P(nt, ns);
		P.reserve(ns);
		std::vector<Eigen::Triplet<double>> ptriplets;
		ptriplets.reserve(ns);

		corres = nearestNeighbors(
			(trgEvecs.block(0, 0, nt, current) * fmap).transpose(),
			srcEvecs.block(0, 0, ns, current).transpose()
			,trgSegments, srcSegments
		);

		for (int j = 0; j < (int)ns; j++) ptriplets.emplace_back(corres[j], j, 1.0);
		P.setFromTriplets(ptriplets.begin(), ptriplets.end());

		fmap = ratio * trgEvecs.block(0, 0, nt, current+stride).transpose() *
			(P * (srcAreas.asDiagonal() *	srcEvecs.block(0, 0, ns, current+stride)));

		signal = true;
	}

	Eigen::MatrixXd coordManHarmonics(const Eigen::MatrixXd& pos, const Eigen::MatrixXd& evecs, const Eigen::VectorXd& areas) {
		Eigen::MatrixXd error =	pos - evecs * (evecs.transpose() * (areas.asDiagonal() * pos));
		Eigen::MatrixXd cmh = evecs;

		for (size_t i = 0; i < 3; i++) {
			Eigen::VectorXd mcmh = error.col(i) - cmh * (cmh.transpose() * 	(areas.asDiagonal() * error.col(i)));

			mcmh = mcmh.array()/sqrt(mcmh.transpose()* (areas.asDiagonal()* mcmh));

			cmh = eig::hConcatDenseMat(cmh, mcmh);
		}

		return cmh;
	}

	void showMatrix(const Eigen::MatrixXd& mat) {
		std::ofstream temp("matrix.temp");
		for (Eigen::Index i = 0; i < mat.rows(); i++) {
			for (Eigen::Index t = 0; t < mat.cols() - 1; t++) {
				temp << mat(i, t) << " ";
			}
			temp << mat(i, mat.cols() - 1) << "\n";
		}
		temp.close();

		system(("python C:\\CGraphics\\show_matrix.py matrix.temp " + std::to_string(mat.rows()) + " " + std::to_string(mat.cols())).c_str());
		std::filesystem::remove("matrix.temp");
	}

}