#pragma once
#include <random>
#include <numeric>
#include <vector>
#include <Eigen/Core>

namespace cpd {
	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	T pairwiseSquaredNorm(const Eigen::MatrixBase<Derived1>& A,
												const Eigen::MatrixBase<Derived2>& B) {
		assert(A.cols() == B.cols());
		int n = static_cast<int>(A.rows());
		T sum = 0;
		#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < n; i++) {
			sum += ((B.rowwise() - A.row(i)).squaredNorm());
		}
		return sum;
	}

	template <typename Derived, typename T = typename Derived::Scalar>
	void replaceInf2ValidNum(Eigen::MatrixBase<Derived>& A, T val) {
		A = A.array().unaryExpr([val](T v) { return std::isfinite(v) ? v : val; });
	}

	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::Matrix<T, -1, -1, O> vConcatDenseMat(const Eigen::DenseBase<Derived1>& A,
																							const Eigen::DenseBase<Derived2>& B) {
		assert(A.cols() == B.cols());
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O> C(A.rows() + B.rows(), A.cols());
		C << A, B;
		return C;
	}

	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T1 = std::vector<size_t>, typename T2 = typename Derived::Scalar>
	Eigen::Matrix<T2, -1, -1, O> sortByIndices(const Eigen::MatrixBase<Derived>& A, const T1& indices,
																						 const bool inverse = false, const int axis = -1) {
		Eigen::VectorXi I(indices.size());
		for (size_t i = 0; i < indices.size(); i++) I[i] = static_cast<int>(indices[i]);
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(I);
		if (inverse) perm = perm.transpose();

		if (axis == 0) {
			assert(A.cols() == indices.size());
			return A * perm;
		} else if (axis == 1) {
			assert(A.rows() == indices.size());
			return perm.transpose() * A;
		}
		assert((A.cols() == indices.size()) && (A.rows() == indices.size()));
		return perm.transpose() * A * perm;
	}

	std::vector<size_t> randomPermutation(const size_t k) {
		std::random_device seed;
		std::mt19937 engine(seed());

		std::vector<size_t> randomIndices(k);
		std::iota(randomIndices.begin(), randomIndices.end(), 0);
		std::shuffle(randomIndices.begin(), randomIndices.end(), engine);

		return randomIndices;
	}

	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T1 = std::vector<size_t>, typename T2 = typename Derived::Scalar>
	Eigen::Matrix<T2, -1, -1, O> randomSampling(const Eigen::MatrixBase<Derived>& A, const size_t k, T1& indices) {
		indices = randomPermutation(A.rows());
		return sortByIndices<O>(A, indices, false, 1).block(0, 0, k, A.cols());
	}

	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T1 = std::vector<size_t>, typename T2 = typename Derived::Scalar>
	Eigen::Matrix<T2, -1, -1, O> randomSampling(const Eigen::MatrixBase<Derived>& A, const size_t k) {
		std::vector<size_t> indices;
		return randomSampling<O>(A, k, indices);
	}

	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T = typename Derived::Scalar>
	std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O>, Eigen::Matrix<T, Eigen::Dynamic, 1>> nystromApprox(const Eigen::MatrixBase<Derived>& C) {
		using Mat = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O>;
		Mat W = C.block(0, 0, C.cols(), C.cols());

		Eigen::JacobiSVD<Mat> SVD(W, Eigen::ComputeThinV);
		auto V = SVD.matrixV();

		Eigen::Matrix<T, Eigen::Dynamic, 1> D = SVD.singularValues();
		Mat U = C * (sqrt((T)C.cols() / C.rows()) * V * D.asDiagonal().inverse());
		D = ((T)C.rows() / C.cols())*D;

		return std::make_tuple(U, D);
	}

}