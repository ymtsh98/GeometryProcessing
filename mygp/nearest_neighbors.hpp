#pragma once
namespace gp::nn {
	std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg) {
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

			(src.colwise() - trg.col(i)).colwise().norm().minCoeff(&minIndex);
			nn[i] = minIndex;
		}
		return nn;
	}

	std::pair<std::vector<int>, std::vector<double>> nearestNeighborsDist(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg) {
		std::vector<int> nn(trg.cols(), 0);
		std::vector<double> dist(trg.cols());

		#pragma omp parallel for num_threads(6)
		for (int i = 0; i < trg.cols(); i++) {
			int minIndex = 0;

			dist[i] = (src.colwise() - trg.col(i)).colwise().norm().minCoeff(&minIndex);
			nn[i] = minIndex;
		}
		return std::make_pair(nn, dist);
	}

	std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg, 
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

	std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg,
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
}