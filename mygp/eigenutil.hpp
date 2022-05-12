#pragma once
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//OpenMP >= 4.0
//#pragma omp declare reduction (+: Eigen::VectorXd: omp_out=omp_out+omp_in) \
//	initializer(omp_priv=VectorXd::Zero(omp_orig.size()))
//pragma omp parallel for reduction(+:_)

namespace gp::eig {

	// pairwise squared norm 
	// @param A MxK(type=0), KxM(type=1) matrix
	// @param B NxK(type=0), KxN(type=1) matrix
	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	T pairwiseSquaredNorm(const Eigen::MatrixBase<Derived1>& A,
												const Eigen::MatrixBase<Derived2>& B, 
		const int type = 0) {
		T sum = 0;
		if (type == 0) {
			assert(A.cols() == B.cols());
			int n = static_cast<int>(A.rows());
			#pragma omp parallel for reduction(+:sum)
			for (int i = 0; i < n; i++) {
				sum += ((B.rowwise() - A.row(i)).squaredNorm());
			}

		} else if (type == 1) {
			assert(A.rows() == B.rows());
			int n = static_cast<int>(A.cols());
			#pragma omp parallel for reduction(+:sum)
			for (int i = 0; i < n; i++) {
				sum += ((B.colwise() - A.col(i)).squaredNorm());
			}

		}
		return sum;
	}

	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	T pairwiseSquaredNormNaive(const Eigen::MatrixBase<Derived1>& A,
														 const Eigen::MatrixBase<Derived2>& B) {
		assert(A.cols() == B.cols());
		int n = A.rows(), m = B.rows();
		T sum = 0;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				sum += (A.row(i) - B.row(j)).squaredNorm();

		return sum;
	}

	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	T pairwiseSquaredNormOneliner(const Eigen::MatrixBase<Derived1>& A,
																const Eigen::MatrixBase<Derived2>& B) {
		assert(A.cols() == B.cols());
		return B.rows() * A.squaredNorm() + A.rows() * B.squaredNorm() - 2 * (A * B.transpose()).sum();
	}


	template <typename Derived, typename T = typename Derived::Scalar>
	void replaceInf2ValidNum(Eigen::MatrixBase<Derived>& A, T val) {
		A = A.array().unaryExpr([val](T v) { return std::isfinite(v) ? v : val; });
	}

	//vertically : Dense + Dense
	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::Matrix<T, -1, -1, O> vConcatDenseMat(const Eigen::DenseBase<Derived1>& A,
																							const Eigen::DenseBase<Derived2>& B) {
		assert(A.cols() == B.cols());
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O> C(A.rows() + B.rows(), A.cols());
		C << A, B;
		return C;
	}

	//horizontally : Dense + Dense
	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::Matrix<T, -1, -1, O> hConcatDenseMat(const Eigen::DenseBase<Derived1>& A,
																							const Eigen::DenseBase<Derived2>& B) {
		assert(A.rows() == B.rows());
		Eigen::Matrix<T, -1, -1, O> C(A.rows(), A.cols() + B.cols());
		C << A, B;
		return C;
	}

	//vertically : Sparse + Sparse
	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::SparseMatrix<T> vConcatSparseMat(const Eigen::SparseCompressedBase<Derived1>& A,
												const Eigen::SparseCompressedBase<Derived2>& B) {
		assert(A.cols() == B.cols());
	
		Eigen::SparseMatrix<T> C(A.rows() + B.rows(), A.cols());
		C.reserve(A.nonZeros() + B.nonZeros());
	
		for (Eigen::Index i = 0; i < A.cols(); i++) {
			C.startVec(i);
			for (typename Eigen::SparseMatrix<T>::InnerIterator itr(A, i); itr; ++itr)
				C.insertBack(itr.row(), i) = itr.value();
			for (typename Eigen::SparseMatrix<T>::InnerIterator itr(B, i); itr; ++itr)
				C.insertBack(A.rows() + itr.row(), i) = itr.value();
		}
	
		C.finalize();
		return C;
	}

	//horizontally : Sparse + Sparse
	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::SparseMatrix<T> hConcatSparseMat(const Eigen::SparseCompressedBase<Derived1>& A,
		const Eigen::SparseCompressedBase<Derived2>& B) {
		assert(A.rows() == B.rows());

		Eigen::SparseMatrix<T> C(A.rows(), A.cols() + B.cols());
		C.reserve(A.nonZeros() + B.nonZeros());

		for (Eigen::Index i = 0; i < A.cols(); i++) {
			C.startVec(i);
			for (typename Eigen::SparseMatrix<T>::InnerIterator itr(A, i); itr; ++itr)
				C.insertBack(itr.row(), i) = itr.value();
		}

		for (Eigen::Index i = 0; i < B.cols(); i++) {
			C.startVec(i + A.cols());
			for (typename Eigen::SparseMatrix<T>::InnerIterator itr(B, i); itr; ++itr)
				C.insertBack(itr.row(), i + A.cols()) = itr.value();
		}

		C.finalize();
		return C;
	}

	//vertically : Sparse + Diagnoal
	template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
	Eigen::SparseMatrix<T> vConcatSparseMat(const Eigen::SparseCompressedBase<Derived1>& A,
																					const Eigen::DenseBase<Derived2>& D) {
		assert(A.cols() == D.rows());
		Eigen::SparseMatrix<T> C(A.rows() + D.rows(), A.cols());
		C.reserve(A.nonZeros() + D.rows());
		for (Eigen::Index i = 0; i < A.cols(); i++) {
			C.startVec(i);
			for (typename Eigen::SparseMatrix<T>::InnerIterator itr(A, i); itr; ++itr)
				C.insertBack(itr.row(), i) = itr.value();
			C.insertBack(A.rows() + i, i) = D[i];
		}
		C.finalize();
		return C;
	}

	//UNDONE : type == 0: indices.size() == cols, type == 1: indices.size() == rows
	template <typename T, typename F = double>
	Eigen::SparseMatrix<F> extractMatrix(const int rows, const int cols, const T& indices, const int type = 0) {
		Eigen::SparseMatrix<F> P(rows, cols);
		P.reserve(cols);
		std::vector<Eigen::Triplet<F>> ptriplets;
		ptriplets.reserve(cols);

		if (type == 0) for (int i = 0; i < cols; i++) ptriplets.emplace_back(indices[i], i, 1.0);
		else if (type == 1) for (int i = 0; i < rows; i++) ptriplets.emplace_back(i, indices[i], 1.0);
		P.setFromTriplets(ptriplets.begin(), ptriplets.end());

		return P;
	}

	//axis = 0 sort cols, axis = 1 sort rows, axis = -1 sort cols and rows
	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T1 = std::vector<int>, typename T2 = typename Derived::Scalar>
	Eigen::Matrix<T2, -1, -1, O> sortByIndices(const Eigen::MatrixBase<Derived>& A, const T1& indices,
																						 const bool inverse = false, const int axis = -1) {
		Eigen::VectorXi I(indices.size());
		for (size_t i = 0; i < indices.size(); i++) I[i] = indices[i];
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

	//axis = 0 sort cols, axis = 1 sort rows, axis = -1 sort cols and rows
	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T1 = Eigen::VectorXd, typename T2 = typename Derived::Scalar>
	Eigen::Matrix<T2, -1, -1, O> sortByWeights(const Eigen::MatrixBase<Derived>& A, const T1& weights, const int axis = 0) {
		std::vector<int> index(weights.size());
		std::iota(index.begin(), index.end(), 0);
		std::sort(index.begin(), index.end(), [&weights](auto& l, auto& r) { return weights[l] < weights[r]; });
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(Eigen::Map<Eigen::VectorXi>(index.data(), index.size()));

		if (axis == 0) {
			assert(A.cols() == weights.size());
			return A * perm;
		} else if (axis == 1) {
			assert(A.rows() == weights.size());
			return perm * A;
		}
		assert((A.cols() == weights.size()) && (A.rows() == weights.size()));
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

	// @param C NxK matrix (K = the number of samples)
	template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived, typename T = typename Derived::Scalar>
	std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O>, Eigen::Matrix<T, Eigen::Dynamic, 1>> nystromApprox(const Eigen::MatrixBase<Derived>& C) {
		using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O>;
		Mat W = C.block(0, 0, C.cols(), C.cols());

		Eigen::JacobiSVD<Mat> SVD(W, Eigen::ComputeThinV);
		auto V = SVD.matrixV();

		Eigen::Matrix<T, Eigen::Dynamic, 1> D = SVD.singularValues();
		Mat U = C * (sqrt((T)C.cols() / C.rows()) * V * D.asDiagonal().inverse());
		D = ((T)C.rows() / C.cols())*D;

		return std::make_tuple(U, D);
	}

	/// @return rows, cols, vals
	template <typename Derived, typename T = typename Derived::Scalar>
	std::tuple<std::vector<Eigen::Index>, std::vector<Eigen::Index>, std::vector<T>> 
		getDataFromSparseMat(const Eigen::SparseCompressedBase<Derived>& A, const Eigen::Index offset = 0) {
		std::vector<Eigen::Index> rows, cols;
		std::vector<T> vals;
		auto nnz = A.nonZeros();
		rows.reserve(nnz);
		cols.reserve(nnz);
		vals.reserve(nnz);
		for (Eigen::Index k = 0; k < A.outerSize(); k++) {
			for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
				rows.emplace_back(it.row() + offset);
				cols.emplace_back(it.col() + offset);
				vals.emplace_back(it.value());
			}
		}

		return std::make_tuple(std::move(rows), std::move(cols), std::move(vals));
	}

	/// @return rows, cols, vals
	template <typename Derived, typename T = typename Derived::Scalar>
	std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<T>>
		getDataFromSparseMatUint(const Eigen::SparseCompressedBase<Derived>& A, const size_t offset = 0) {
		std::vector<size_t> rows, cols;
		std::vector<T> vals;
		auto nnz = A.nonZeros();
		rows.reserve(nnz);
		cols.reserve(nnz);
		vals.reserve(nnz);
		for (Eigen::Index k = 0; k < A.outerSize(); k++) {
			for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
				rows.emplace_back((size_t)it.row() + offset);
				cols.emplace_back((size_t)it.col() + offset);
				vals.emplace_back(it.value());
			}
		}

		return std::make_tuple(std::move(rows), std::move(cols), std::move(vals));
	}

	template <typename T, typename F = typename T::value_type>
	Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic> vec2eig(const std::vector<T>& vec) {
		Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic> eig(vec.size(), vec[0].size());
		for (size_t i = 0; i < vec.size(); i++) {
			for (size_t j = 0; j < vec[0].size(); j++) eig(i, j) = vec[i][j];
		}
		return eig;
	}

	template <typename T = double>
	Eigen::SparseMatrix<T> spIdentity(const size_t rows, const size_t cols) {
		Eigen::SparseMatrix<T> eye(rows, cols);
		eye.setIdentity();
		return eye;
	}

	template <typename T = double>
	Eigen::SparseMatrix<T> spIdentity(const size_t size) {
		return spIdentity(size, size);
	}

	template <typename Scalar1, typename Scalar2>
	void append(Eigen::Vector<Scalar1, -1>& vector, const Scalar2 value) {
		vector.conservativeResize(vector.rows() + 1);
		vector(vector.rows() - 1) = static_cast<Scalar1>(value);
	}

	Eigen::MatrixXd expandMatWithEye(const Eigen::MatrixXd& A, const Eigen::Index rows, const Eigen::Index cols) {
		assert(rows > A.rows() && cols > A.cols());
		Eigen::MatrixXd B = Eigen::MatrixXd::Identity(rows, cols);
		B.block(0, 0, A.rows(), A.cols()) = A;
		return B;
	}

	template<class Matrix>
	void writeBinary(std::string_view filename, const Matrix& matrix) {
		std::ofstream out(filename.data(), std::ios::out | std::ios::binary | std::ios::trunc);
		typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
		out.write((char*)(&rows), sizeof(typename Matrix::Index));
		out.write((char*)(&cols), sizeof(typename Matrix::Index));
		out.write((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
		out.close();
	}

	template<class Matrix>
	void readBinary(std::string_view filename, Matrix& matrix) {
		std::ifstream in(filename, std::ios::in | std::ios::binary);
		typename Matrix::Index rows=0, cols=0;
		in.read((char*)(&rows), sizeof(typename Matrix::Index));
		in.read((char*)(&cols), sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.read((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
		in.close();
	}
}