auto estimate(const Eigen::MatrixXd& Yt, std::reference_wrapper<Eigen::MatrixXd> X, const double variance) {
	double omega = 0.0;
	int n = X.get().rows(), m = Yt.rows();
	int dim = X.get().cols();
	double sigma = 2.0 * variance;
	double c = pow(std::numbers::pi * sigma, dim/2.0) * (omega / (1.0 - omega)) * ((double)Yt.rows() / n);

	Eigen::MatrixXd PX = Eigen::MatrixXd::Zero(m, dim);
	Eigen::VectorXd P1 = Eigen::VectorXd::Zero(m);
	Eigen::VectorXd PT1 = Eigen::VectorXd::Zero(n);
	Eigen::MatrixXd P(m, n);

	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		Eigen::VectorXd e = Eigen::exp(-(Yt.rowwise() - X.get().row(i)).rowwise().squaredNorm().array() / sigma);
		e.array() /= e.sum() + c;
		P.col(i) = e;

		Eigen::MatrixXd PXt = e * X.get().row(i);

		PT1(i) = e.sum();
		#pragma omp critical
		{
			P1 += e;
			PX += PXt;
		}
	}

	double np = P1.sum();
	return P;
	//return std::make_tuple( PX, P1, PT1, np);
}

class Ifgt {
	/*
		Ifgt fgt1(Y, k, 30), fgt2(X, k, 30);

	auto KT1 = fgt1.compute(X, Eigen::VectorXd::Ones(X.rows()), sqrt(2.0 * var));

	Eigen::VectorXd a = 1.0 / (KT1.array() + 0.0);
	Eigen::VectorXd PT1 = 1.0 - a.array();

	auto P1 = fgt2.compute(Y, a, sqrt(2.0 * var));
	*/
	using RowMatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	std::reference_wrapper<const Eigen::MatrixXd> X;
	size_t n, d, k, p;
	Eigen::MatrixXd C;
	std::vector<size_t> clusterIdx;
	std::vector<Eigen::MatrixXd> Cka;
	std::vector<double> c;

public:
	Ifgt(const Eigen::MatrixXd& trg, const size_t k, const size_t p) :
		X(trg), n(trg.rows()), d(trg.cols()), k(k), p(p), clusterIdx(n), Cka(k, Eigen::MatrixXd::Zero(p, d)), c(p) {

		RowMatXf Xr(X.get().cast<float>().leftCols(X.get().cols()));
		RowMatXf Cr(k, d);

		faiss::kmeans_clustering(d, n, k, Xr.data(), Cr.data());

		C = Cr.cast<double>();

		for (size_t i = 0; i < n; i++) {
			size_t j;
			(C.rowwise() - X.get().row(i)).rowwise().squaredNorm().minCoeff(&j);
			clusterIdx[i] = j;
		}

		double afuct = 1.0, asum = 0.0;
		for (size_t i = 0; i < p; i++) {
			asum += i;
			afuct *= std::tgamma(i+1);
			c[i] = pow(2, asum) / afuct;
		}
	}

	Eigen::VectorXd compute(const Eigen::MatrixXd& src, const Eigen::VectorXd& weights, const double h) {
		double h2 = h * h;

		for (size_t i = 0; i < n; i++) {
			size_t j = clusterIdx[i];
			Eigen::VectorXd xc = X.get().row(i) - C.row(j);
			double xcSqNorm = xc.squaredNorm() / h2;
			xc.array() /= h;

			for (size_t a = 0; a < p; a++) {
				Cka[j].row(a).array() += c[a] * weights[i] * exp(-xcSqNorm) * Eigen::pow(xc.array(), a);
			}
		}

		Eigen::VectorXd G = Eigen::VectorXd::Zero(src.rows());
		for (size_t i = 0; i < src.rows(); i++) {
			for (size_t j = 0; j < k; j++) {
				Eigen::VectorXd sc = src.row(i) - C.row(j);
				double scSqNorm = sc.squaredNorm() / h2;
				sc.array() /= h;

				for (size_t a = 0; a < p; a++) {
					G(i) += exp(-scSqNorm) * Cka[j].row(a) * Eigen::VectorXd(Eigen::pow(sc.array(), a));
				}
			}
		}
		return G;
	}

};