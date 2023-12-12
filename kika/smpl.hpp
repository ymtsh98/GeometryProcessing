#include <LBFGS.h>

namespace kika::smpl {

class SpatialRegressor {
private:
	size_t q, n;

	Eigen::MatrixX3d X;
	Eigen::MatrixX3d J;
	Eigen::MatrixXd F;
	Eigen::MatrixXd N;
	Eigen::SparseMatrix<double> S;

	double coeffRecon, coeffLocal, coeffSparse, coeffConvex, coeffLock;

public:
	SpatialRegressor(
		const Eigen::MatrixX3d& pos, const Eigen::MatrixX3d& joints,
		const Eigen::MatrixXd& weights, const std::vector<int>& parents,
		const double coeffRecon = 1., const double coeffLocal = 1000.,
		const double coeffSparse = 100., const double coeffConvex = 10., const double coeffLock = 1.0) : q(q), n(n), X(pos), J(joints),
		coeffRecon(coeffRecon), coeffLocal(coeffLocal), coeffSparse(coeffSparse), coeffConvex(coeffConvex), coeffLock(coeffLock){

		double eps = 0.0001;

		q = (size_t)joints.rows();
		n = (size_t)pos.rows();

		Eigen::MatrixXd knnWeights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
		kika::knn::NanoFlannKNN vertexTree(pos);
		std::vector<std::vector<size_t>> knn;
		std::vector<std::vector<float>> dists;
		vertexTree.getKNearestNeighbors(joints, pos.rows() * 0.1, knn, dists);
		for (size_t i = 0; i < (size_t)joints.rows(); i++) {
			knnWeights(knn[i], i).array() = 1.0;
		}

		F = Eigen::MatrixXd::Zero(q, n);
		N = Eigen::MatrixXd::Zero(q, n);

		for (size_t i = 0; i < q; i++) {
			double sum = 0.0;
			int p = parents[i] == -1 ? i : parents[i];
			Eigen::VectorXd wp = weights.col(p);
			Eigen::VectorXd wi = weights.col(i);

			Eigen::VectorXd w = wi.array() * wp.array();
			F.row(i) = (w.array() > eps).select(1, w);
			sum = F.row(i).sum();

			if (sum < 1) {
				w = knnWeights.col(i).array() * weights.col(p).array();
				F.row(i) = (w.array() > eps).select(1, w);
				sum = F.row(i).sum();
			}

			if (sum < 1) {
				w = knnWeights.col(i).array() * knnWeights.col(p).array();
				F.row(i) = (w.array() > eps).select(1, w);
				sum = F.row(i).sum();
			}

			if (sum < 1) {
				w = knnWeights.col(i);
				F.row(i) = (w.array() > eps).select(1, w);
				sum = F.row(i).sum();
			}
		}

		kika::knn::NanoFlannKNN tree(joints);
		tree.getKNearestNeighbors(joints, 2, knn, dists);

		S.resize(q, q);
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(q);
		for (size_t i = 0; i < knn.size(); i++) {
			if (dists[i][1] < 1e-8) {
				triplets.emplace_back((int)i, (int)knn[i][1], -1.);
			}
			triplets.emplace_back((int)i, (int)i, 1.);
		}


	}

	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
		double fx = 0.0;

		Eigen::MatrixXd R = x.reshaped(q, n);
		Eigen::MatrixXd gradR = Eigen::MatrixXd::Zero(q, n);

		fx += coeffRecon * (R * X - J).squaredNorm();
		gradR += 2. * coeffRecon * (R * X - J) * X.transpose();

		fx += coeffLocal * ((F.array() - 1) * R.array()).square().sum();
		gradR.array() += 2. * coeffLocal * ((F.array() - 1).square() * R.array());

		fx += coeffConvex * (R * Eigen::VectorXd::Ones(n) - Eigen::VectorXd::Ones(q)).squaredNorm();
		gradR += 2. * coeffConvex * (R * Eigen::VectorXd::Ones(n) - Eigen::VectorXd::Ones(q)) * Eigen::VectorXd::Ones(n).transpose();

		grad = Eigen::Map<Eigen::VectorXd>(gradR.data(), q * n);
		return fx;
	}
};

inline Eigen::MatrixXd jointRegressor(
	const Eigen::MatrixX3d& pos, const Eigen::MatrixX3d& joints,
	Eigen::MatrixXd weights, const std::vector<int>& parents,
	const size_t iter = 1000) {

	SpatialRegressor spreg(pos, joints, weights, parents);

	LBFGSpp::LBFGSParam<double> param;
	param.max_iterations = iter;
	LBFGSpp::LBFGSSolver<double> solver(param);

	Eigen::VectorXd flatR = Eigen::VectorXd::Zero(weights.rows() * weights.cols(), 1);

	double fx;
	int niter = solver.minimize(spreg, flatR, fx);
	return flatR.reshaped(weights.cols(), weights.rows());
}

}