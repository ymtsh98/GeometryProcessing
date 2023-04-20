#pragma once
#include <vector>
#include <Eigen/Core>

namespace gp::nn {
#ifndef MYGP_IMPLEMENTATION
std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg);

std::vector<int> nearestNeighborsCmplx(const Eigen::MatrixXcd& src, const Eigen::MatrixXcd& trg);

double chamferDistance(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg);

std::vector<int> nearestNeighbors(const std::vector<std::vector<uint32_t>>& srcNeighbs,
	const std::vector<std::vector<uint32_t>>& trgNeighbs,
	const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg);

std::pair<Eigen::VectorXi, Eigen::VectorXd> nearestNeighborsDist(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg);
std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg,
	const Eigen::Vector<uint32_t, -1>& srcSeg, const Eigen::Vector<uint32_t, -1>& trgSeg);

std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg,
	const Eigen::MatrixXd& exsrc, const Eigen::MatrixXd& extrg,
	const Eigen::Vector<uint32_t, -1>& srcSeg, const Eigen::Vector<uint32_t, -1>& trgSeg,
	const double excoeff = 0.2);
#else
inline std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg) {
	std::vector<int> nn(trg.cols(), 0);

	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < trg.cols(); i++) {
		int minIndex = 0;

		//Eigen::VectorXd&& rtrg = trg.col(i);
		//double minDist = std::numeric_limits<double>::max();
		//for (int t = 0; t < src.cols(); t++) {
		//	double dist = (rtrg - src.col(t)).norm();
		//
		//	if (minDist > dist) {
		//		minDist = dist;
		//		minIndex = t;
		//	}
		//}

		(src.colwise() - trg.col(i)).colwise().squaredNorm().minCoeff(&minIndex);
		nn[i] = minIndex;
	}
	return nn;
}

inline std::vector<int> nearestNeighborsCmplx(const Eigen::MatrixXcd& src, const Eigen::MatrixXcd& trg) {
	std::vector<int> nn(trg.cols(), 0);

	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < trg.cols(); i++) {
		int minIndex = 0;

		(src.colwise() - trg.col(i)).colwise().squaredNorm().minCoeff(&minIndex);
		nn[i] = minIndex;
	}
	return nn;
}

inline double chamferDistance(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg) {
	double d = 0.0;

	for (int i = 0; i < src.cols(); i++) {
		d += (trg.colwise() - src.col(i)).colwise().squaredNorm().minCoeff();
	}

	for (int i = 0; i < trg.cols(); i++) {
		d += (src.colwise() - trg.col(i)).colwise().squaredNorm().minCoeff();
	}

	return d;
}

inline std::vector<int> nearestNeighbors(const std::vector<std::vector<uint32_t>>& srcNeighbs,
	const std::vector<std::vector<uint32_t>>& trgNeighbs, 
	const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg) {
	std::vector<int> nn(trg.cols(), 0);

	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < trg.cols(); i++) {
		int minIndex = 0;

		Eigen::MatrixXd rtrg = trg(Eigen::all, trgNeighbs[i]);
		double minDist = std::numeric_limits<double>::max();
		for (int t = 0; t < src.cols(); t++) {
			double dist = (rtrg.colwise() - src.col(t)).colwise().squaredNorm().maxCoeff();
			
			if (minDist > dist) {
				minDist = dist;
				minIndex = t;
			}
		}

		nn[i] = minIndex;
	}
	return nn;
}

inline std::pair<Eigen::VectorXi, Eigen::VectorXd> nearestNeighborsDist(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg) {
	Eigen::VectorXi nn(trg.cols());
	Eigen::VectorXd dist(trg.cols());

	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < dist.size(); i++) {
		int minIndex = 0;

		dist[i] = (src.colwise() - trg.col(i)).colwise().norm().minCoeff(&minIndex);
		nn[i] = minIndex;
	}
	return std::make_pair(nn, dist);
}

inline std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg,
	const Eigen::Vector<uint32_t, -1>& srcSeg, const Eigen::Vector<uint32_t, -1>& trgSeg) {
	std::vector<int> nn(trg.cols(), 0);

	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < trg.cols(); i++) {
		int minIndex = 0;

		Eigen::VectorXd&& rtrg = trg.col(i);
		double minDist = std::numeric_limits<double>::max();
		for (int t = 0; t < src.cols(); t++) {
			double region = 1.0 / (1.0 + std::popcount<uint32_t>(trgSeg[i]&srcSeg[t]));
			double dist = region * (rtrg - src.col(t)).squaredNorm();
			
			if (minDist > dist) {
				minDist = dist;
				minIndex = t;
			}
		}

		//(src.colwise() - trg.col(i)).colwise().norm().minCoeff(&minIndex);
		nn[i] = minIndex;
	}

	return nn;
}

inline std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg,
	const Eigen::MatrixXd& exsrc, const Eigen::MatrixXd& extrg,
	const Eigen::Vector<uint32_t, -1>& srcSeg, const Eigen::Vector<uint32_t, -1>& trgSeg, 
	const double excoeff = 0.2) {
	std::vector<int> nn(trg.cols(), 0);

	//double avgexdist = eig::pairwiseSquaredNorm(exsrc, extrg, 1) / exsrc.cols() / extrg.cols();

	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < trg.cols(); i++) {
		int minIndex = 0;

		Eigen::VectorXd&& rextrg = extrg.col(i);
		Eigen::VectorXd&& rtrg = trg.col(i);
		double minDist = std::numeric_limits<double>::max();
		for (int t = 0; t < src.cols(); t++) {
			double region = 1.0 / (1.0 + std::popcount<uint32_t>(trgSeg[i]&srcSeg[t]));
			double exdist = (rextrg - exsrc.col(t)).squaredNorm();
			double dist = region * ((rtrg - src.col(t)).squaredNorm() + excoeff * exdist);

			if (minDist > dist) {
				minDist = dist;
				minIndex = t;
			}
		}

		nn[i] = minIndex;
	}

	return nn;
}
#endif
}