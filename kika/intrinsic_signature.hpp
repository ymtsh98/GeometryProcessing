#pragma once

namespace kika::sign {


class HeatKernel {
	const Eigen::VectorXd& evals;
	const Eigen::MatrixXd& evecs;

public:
	HeatKernel(const Eigen::VectorXd& evals, const Eigen::MatrixXd& evecs) : evals(evals), evecs(evecs) {}

	// prepare heat kernel signature parameters
	Eigen::VectorXd parameters(const size_t n) {
		Eigen::VectorXd params(n);
		double minParam = abs(4.0 * log(10.) / evals[evals.size() - 1]);
		double maxParam = abs(4.0 * log(10.) / evals[1]);
		double step = log(maxParam) - log(minParam);

		for (size_t i = 0; i < n; i++) {
			params[i] = exp(log(minParam) + step * (i + 1) / (double)n);
		}

		return params;
	}

	// compute hks
	// @return |V| x n matrix
	Eigen::MatrixXd compute(const size_t n) {
		Eigen::MatrixXd hks(evecs.rows(), n);

		size_t numeigs = evals.size();
		auto energy = parameters(n);

		Eigen::MatrixXd phi2 = evecs.block(0, 1, evecs.rows(), numeigs - 1).array().square();
		Eigen::VectorXd E = evals.block(1, 0, numeigs - 1, 1);
		#pragma omp parallel for
		for (int i = 0; i < (int)n; i++) {
			Eigen::VectorXd C = Eigen::exp(-energy[i] * E.array());
			hks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}

		return hks;
	}

	// compute hks with landmark
	// @return |V| x n matrix
	Eigen::MatrixXd compute(const size_t n, const size_t landmark) {
		Eigen::MatrixXd hks(evecs.rows(), n);

		size_t numeigs = evals.size();
		auto energy = parameters(n);

		Eigen::MatrixXd lm = evecs.row(landmark).replicate(evecs.rows(), 1);
		Eigen::MatrixXd phi2 = (lm.array() * evecs.array()).block(0, 1, evecs.rows(), numeigs - 1);
		Eigen::VectorXd E = evals.block(1, 0, numeigs - 1, 1);

		#pragma omp parallel for
		for (int i = 0; i < (int)n; i++) {
			Eigen::VectorXd C = Eigen::exp(-energy[i] * E.array());
			hks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}

		return hks;
	}

};

class WaveKernel {
	const Eigen::VectorXd& evals;
	const Eigen::MatrixXd& evecs;

public:
	WaveKernel(const Eigen::VectorXd& evals, const Eigen::MatrixXd& evecs) : evals(evals), evecs(evecs) {}

	// prepare wave kernel signature parameters
	std::pair<Eigen::VectorXd, double> parameters(const size_t n) {
		Eigen::VectorXd params(n);
		double minParam = log(evals[evals.size()-1]) / 1.02;
		double maxParam = log(evals[1]);
		double step = maxParam - minParam;

		for (size_t i = 0; i < n; i++) {
			params[i] = minParam + step * (i + 1) / (double)n;
		}

		double sigma = n > 0 ? (params[1] - params[0]) * 5.0 : 0.0;

		return { params, sigma };
	}

	// compute wks
	// @return |V| x n matrix
	Eigen::MatrixXd compute(const size_t n) {
		Eigen::MatrixXd wks(evecs.rows(), n);

		size_t numeigs = evals.size();
		auto [energy, sigma] = parameters(n);
		double s = 2.0 * sigma * sigma;

		Eigen::MatrixXd phi2 = evecs.block(0, 1, evecs.rows(), numeigs - 1).array().square();
		Eigen::VectorXd E = Eigen::log(evals.array()).block(1, 0, numeigs - 1, 1);

		#pragma omp parallel for
		for (int i = 0; i < (int)n; i++) {
			Eigen::VectorXd C = Eigen::exp(-(Eigen::VectorXd::Constant(numeigs - 1, energy[i]).array() - E.array()).square() / s);
			wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}

		return wks;
	}

	// compute wks with landmark
	// @return |V| x n matrix
	Eigen::MatrixXd compute(const size_t n, const size_t landmark) {
		Eigen::MatrixXd wks(evecs.rows(), n);

		size_t numeigs = evals.size();
		auto [energy, sigma] = parameters(n);
		double s = 2.0 * sigma * sigma;

		Eigen::MatrixXd lm = evecs.row(landmark).replicate(evecs.rows(), 1);
		Eigen::MatrixXd phi2 = (lm.array() * evecs.array()).block(0, 1, evecs.rows(), numeigs - 1);
		Eigen::VectorXd E = Eigen::log(evals.array()).block(1, 0, numeigs - 1, 1);

		#pragma omp parallel for
		for (int i = 0; i < (int)n; i++) {
			Eigen::VectorXd C = Eigen::exp(-(Eigen::VectorXd::Constant(numeigs - 1, energy[i]).array() - E.array()).square() / s);
			wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}

		return wks;
	}

};


}