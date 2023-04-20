#pragma once
#include <vector>
#include <Eigen/Core>

namespace cpd {
	template <typename Derived, typename T = typename Derived::Scalar>
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> gaussianKernel(
		const Eigen::MatrixBase<Derived>& V, const T beta) {

		int n = static_cast<int>(V.rows());
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> kernel(n, n);

		const T b = 2.0 * beta * beta;
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			kernel.col(i) = Eigen::exp(-(V.rowwise() - V.row(i)).rowwise().squaredNorm().array() / b);
		}
		return kernel;
	}

	template <typename Derived, typename T = typename Derived::Scalar, typename I = size_t>
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> gaussianKernel(
		const Eigen::MatrixBase<Derived>& V, const T beta, const std::vector<I>& indices) {

		int k = static_cast<int>(indices.size());
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> kernel(V.rows(), k);

		const T b = 2.0 * beta * beta;
		#pragma omp parallel for
		for (int i = 0; i < k; i++) {
			kernel.col(i) = Eigen::exp(-(V.rowwise() - V.row(indices[i])).rowwise().squaredNorm().array() / b);
		}
		return kernel;
	}

	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> gaussianKernel(
		const Eigen::MatrixBase<Derived1>& V, const Eigen::MatrixBase<Derived2>& U, const T beta) {

		int n = static_cast<int>(V.rows()), m = static_cast<int>(U.rows());
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> kernel(n, m);

		const T b = 2.0 * beta * beta;
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			kernel.col(i) = Eigen::exp(-(V.rowwise() - U.row(i)).rowwise().squaredNorm().array() / b);
		}
		return kernel;
	}

}