#pragma once
namespace gp {
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
				double dist = region * (rtrg - src.col(t)).norm();
			
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

	//std::vector<int> nearestNeighbors(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg,
	//																	std::vector<size_t>& src_segment, std::vector<size_t>& trg_segment,
	//																	std::set<std::pair<size_t, size_t>>& adjacent, int ad) {
	//	std::vector<int> correspondence(trg.size(), 0);

	//	#pragma omp parallel for
	//	for (int i = 0; i < trg.cols(); i++) {
	//		int min_index = 0;
	//		Eigen::VectorXd&& rtrg = trg.col(i);

	//		size_t ts = trg_segment[i];
	//		double min_dist = std::numeric_limits<double>::max();
	//		for (int t = 0; t < src.cols(); t++) {
	//			double dist = (rtrg - src.col(t)).norm();

	//			bool test = false;
	//			if (ad == 0) test = true;
	//			else if (ad == 1) test = true;
	//			else if (ad == 2) test = adjacent.contains({ ts, src_segment[t] });
	//			if (min_dist > dist && test) {
	//				min_dist = dist;
	//				min_index = t;
	//			}
	//		}

	//		//(src.colwise() - rtrg).colwise().norm().minCoeff(&min_index);

	//		#pragma omp atomic
	//		correspondence[i] += min_index;
	//	}
	//	return correspondence;
	//}
}