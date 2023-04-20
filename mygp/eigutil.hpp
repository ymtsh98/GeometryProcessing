#pragma once
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/// 
/// basic utility
/// 
namespace gp::eig {
template<class Matrix>
void writeBinary(const std::string& filename, const Matrix& mat) {
	std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	typename Matrix::Index rows=mat.rows(), cols=mat.cols();
	out.write((char*)(&rows), sizeof(typename Matrix::Index));
	out.write((char*)(&cols), sizeof(typename Matrix::Index));
	out.write((char*)mat.data(), rows*cols*sizeof(typename Matrix::Scalar));
	out.close();
}

template<class Matrix>
void readBinary(const std::string& filename, Matrix& mat) {
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	typename Matrix::Index rows=0, cols=0;
	in.read((char*)(&rows), sizeof(typename Matrix::Index));
	in.read((char*)(&cols), sizeof(typename Matrix::Index));
	mat.resize(rows, cols);
	in.read((char*)mat.data(), rows*cols*sizeof(typename Matrix::Scalar));
	in.close();
}

template <int Dim=Eigen::Dynamic, typename T, typename F = typename T::value_type>
Eigen::Matrix<F, Eigen::Dynamic, Dim> vec2eig(const std::vector<T>& vec) {
	Eigen::Matrix<F, Eigen::Dynamic, Dim> eig(vec.size(), Dim);
	for (size_t i = 0; i < vec.size(); i++) {
		for (int j = 0; j < Dim; j++) {
			eig(i, j) = static_cast<F>(vec[i][j]);
		}
	}
	return eig;
}

template <typename Derived, typename T = typename Derived::Scalar>
void replaceInvalid2Valid(Eigen::MatrixBase<Derived>& A, T val) {
	A = A.array().unaryExpr([val](T v) { return (!std::isfinite(v) || std::isnan(v)) ? v : val; });
}
}

/// 
/// dense matrix extensions
/// 
namespace gp::eig {
inline Eigen::VectorXd zeroOneNorm(const Eigen::VectorXd& vector) {
	Eigen::VectorXd ret = vector;
	ret.array() -= ret.minCoeff();
	ret.array() /= ret.maxCoeff();
	return ret;
}

template <typename Scalar>
void append(Eigen::Vector<Scalar, -1>& vector, const Scalar value) {
	vector.conservativeResize(vector.rows() + 1);
	vector(vector.rows() - 1) = value;
}

template <typename Derived, typename Scalar = Derived::Scalar, int Rows = Derived::RowsAtCompileTime,
	int Cols = Derived::ColsAtCompileTime>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	expandMatWithZero(const Eigen::DenseBase<Derived>& A, const Eigen::Index rows, const Eigen::Index cols) {
	assert(rows >= A.rows() && cols >= A.cols());
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)> B(rows, cols);
	B.fill(0.0);
	B.block(0, 0, A.rows(), A.cols()) = A;
	return B;
}

template <typename Derived, typename Scalar = Derived::Scalar, int Rows = Derived::RowsAtCompileTime, 
	int Cols = Derived::ColsAtCompileTime>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	expandMatWithEye(const Eigen::DenseBase<Derived>& A, const Eigen::Index rows, const Eigen::Index cols) {
	assert(rows >= A.rows() && cols >= A.cols());
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)> B(rows, cols);
	B.fill(0.0);
	B.setIdentity();
	B.block(0, 0, A.rows(), A.cols()) = A;
	return B;
}

template <typename Derived, typename Scalar = Derived::Scalar, int Rows = Derived::RowsAtCompileTime,
	int Cols = Derived::ColsAtCompileTime>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	expandMatWithValue(const Eigen::DenseBase<Derived>& A, const Eigen::Index rows, const Eigen::Index cols, const Scalar value) {
	assert(rows >= A.rows() && cols >= A.cols());
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)> B(rows, cols);
	B.fill(value);
	B.block(0, 0, A.rows(), A.cols()) = A;
	return B;
}

namespace impl {
template <typename Derived, typename... Args>
void concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(const Eigen::DenseBase<Derived>& A, const Args&... args) {
	if constexpr (sizeof...(args) != 0) concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);
}

template <typename T, typename Derived, typename... Args>
void hAppendDenseMatImpl(Eigen::MatrixX<T>& A, const Eigen::Index& offset, const Eigen::DenseBase<Derived>& B, const Args&... args) {
	A.block(0, offset, A.rows(), B.cols()) = B;
	if constexpr (sizeof...(args) != 0) hAppendDenseMatImpl(A, offset + B.cols(), args...);
}

template <typename T, typename Derived, typename... Args>
void vAppendDenseMatImpl(Eigen::MatrixX<T>& A, const Eigen::Index& offset, const Eigen::DenseBase<Derived>& B, const Args&... args) {
	A.block(offset, 0, B.rows(), A.cols()) = B;
	if constexpr (sizeof...(args) != 0) vAppendDenseMatImpl(A, offset + B.rows(), args...);
}

}

//horizontally (row_A, col_A + col_B) : Dense + Dense
template <typename Derived, typename... Args,
	typename Scalar = typename Derived::Scalar, int Rows = Derived::RowsAtCompileTime, int Cols = -1>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	hConcatDenseMat(const Eigen::DenseBase<Derived>& A, const Args&... args) {

	assert(((A.rows() == args.rows()) && ...));
	impl::concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);

	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
		M(A.rows(), A.cols() + (args.cols() + ...));
	((M << A), ..., args);
	return M;
}

//vertically (row_A + row_B, col_A) : Dense + Dense
template <typename Derived, typename... Args,
	typename Scalar = typename Derived::Scalar, int Rows = -1, int Cols = Derived::ColsAtCompileTime>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	vConcatDenseMat(const Eigen::DenseBase<Derived>& A, const Args&... args) {

	assert(((A.cols() == args.cols()) && ...));
	impl::concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);

	Eigen::Matrix<Scalar, Rows, Cols, (Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
		M(A.rows() + (args.rows() + ...), A.cols());
	((M << A), ..., args);
	return M;
}

//horizontally (row_A, col_A + col_B) : Dense + Dense
template <typename T, typename... Args>
void hAppendDenseMat(Eigen::MatrixX<T>& A, const Args&... args) {
	assert(((A.rows() == args.rows()) && ...));
	impl::concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);

	Eigen::Index offset = A.cols();
	A.conservativeResize(Eigen::NoChange, A.cols() + (args.cols() + ...));

	impl::hAppendDenseMatImpl(A, offset, args...);
}

//vertically (row_A + row_B, col_A) : Dense + Dense
template <typename T, typename... Args>
void vAppendDenseMat(Eigen::MatrixX<T>& A, const Args&... args) {
	assert(((A.cols() == args.cols()) && ...));
	impl::concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);

	Eigen::Index offset = A.rows();
	A.conservativeResize(A.rows() + (args.rows() + ...), Eigen::NoChange);

	impl::vAppendDenseMatImpl(A, offset, args...);
}

/*
//horizontally (row_A, col_A + col_B) : Dense + Dense
template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
Eigen::Matrix<T, -1, -1, O> hConcatDenseMat(const Eigen::DenseBase<Derived1>& A,
	const Eigen::DenseBase<Derived2>& B) {
	assert(A.rows() == B.rows());
	Eigen::Matrix<T, -1, -1, O> C(A.rows(), A.cols() + B.cols());
	C << A, B;
	return C;
}

//vertically (row_A + row_B, col_A) : Dense + Dense
template <Eigen::StorageOptions O = Eigen::ColMajor, typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
Eigen::Matrix<T, -1, -1, O> vConcatDenseMat(const Eigen::DenseBase<Derived1>& A,
	const Eigen::DenseBase<Derived2>& B) {
	assert(A.cols() == B.cols());
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, O> C(A.rows() + B.rows(), A.cols());
	C << A, B;
	return C;
}

//horizontally (row_A, col_A + col_B) : Dense + Dense
template <typename T, typename Derived>
void hAppendDenseMat(Eigen::MatrixX<T>& A, const Eigen::DenseBase<Derived>& B) {
	assert(A.rows() == B.rows());
	A.conservativeResize(Eigen::NoChange, A.cols() +  B.cols());
	A.rightCols(B.cols()) = B;
}

//vertically (row_A + row_B, col_A) : Dense + Dense
template <typename T, typename Derived>
void vAppendDenseMat(Eigen::MatrixX<T>& A, const Eigen::DenseBase<Derived>& B) {
	assert(A.cols() == B.cols());
	A.conservativeResize(A.rows() +  B.rows(), Eigen::NoChange);
	A.bottomRows(B.rows()) = B;
}
*/
}

/// 
/// sparse matrix extensions
/// 
namespace gp::eig {
template <typename T = double>
Eigen::SparseMatrix<T> spIdentity(const size_t rows, const size_t cols) {
	Eigen::SparseMatrix<T> eye(rows, cols);
	eye.setIdentity();
	return eye;
}

template <typename T = double>
Eigen::SparseMatrix<T> spIdentity(const size_t size) {
	return spIdentity<T>(size, size);
}

/// random sparse matrix (0.0 - 1.0)
template <typename T=double>
Eigen::SparseMatrix<T> spRandom(const size_t rows, const size_t cols, const double density) {
	std::random_device seed;
	std::mt19937 engine(seed());

	std::uniform_real_distribution<T> dist(0.0, 1.0);

	std::vector<Eigen::Triplet<T>> triplets;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			if (dist(engine) < density) {
				triplets.emplace_back((int)i, (int)j, dist(engine));
			}
		}
	}

	Eigen::SparseMatrix<T> mat(rows, cols);
	mat.setFromTriplets(triplets.begin(), triplets.end());
	return mat;
}

/// @return rows, cols, vals
template <typename I = int, typename Derived, typename T = typename Derived::Scalar>
std::tuple<std::vector<I>, std::vector<I>, std::vector<T>>
getDataFromSparseMat(const Eigen::SparseCompressedBase<Derived>& A, const size_t offset = 0) {
	std::vector<I> rows, cols;
	std::vector<T> vals;
	auto nnz = A.nonZeros();
	rows.reserve(nnz);
	cols.reserve(nnz);
	vals.reserve(nnz);
	for (Eigen::Index k = 0; k < A.outerSize(); k++) {
		for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
			rows.emplace_back(static_cast<I>(it.row() + offset));
			cols.emplace_back(static_cast<I>(it.col() + offset));
			vals.emplace_back(it.value());
		}
	}

	return std::make_tuple(std::move(rows), std::move(cols), std::move(vals));
}

/// @return rows, cols, real, imag
template <typename I = int, typename Derived, typename T = typename Derived::Scalar, typename C = typename T::value_type>
std::tuple<std::vector<I>, std::vector<I>, std::vector<C>, std::vector<C>>
getDataFromSparseMatCmplx(const Eigen::SparseCompressedBase<Derived>& A, const size_t offset = 0) {
	std::vector<I> rows, cols;
	std::vector<C> real, imag;
	auto nnz = A.nonZeros();
	rows.reserve(nnz);
	cols.reserve(nnz);
	real.reserve(nnz);
	imag.reserve(nnz);
	for (Eigen::Index k = 0; k < A.outerSize(); k++) {
		for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
			rows.emplace_back(static_cast<I>(it.row() + offset));
			cols.emplace_back(static_cast<I>(it.col() + offset));
			real.emplace_back(it.value().real());
			imag.emplace_back(it.value().imag());
		}
	}

	return std::make_tuple(std::move(rows), std::move(cols), std::move(real), std::move(imag));
}

//horizontally (row_A, col_A + col_B) : Sparse + Sparse
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

//vertically (row_A + row_B, col_A) : Sparse + Sparse
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

//vertically (row_A + row_B, col_A) : Sparse + Diagnoal
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

// print sparse matrix.
// warnings : row >= 0 && col >= 0 is (or) operation. print elements under row (_) col.
// (or)  print(spmat, row, col)
// (and) print(spmat.topLeftCorner(row, col).eval()).
template <typename Derived>
void print(const Eigen::SparseCompressedBase<Derived>& A, const int row = -1, const int col = -1) {
	if (row == -1 && col == -1) {
		for (int k=0; k < A.outerSize(); ++k) {
			for (typename Eigen::SparseMatrix<typename Derived::Scalar>::InnerIterator it(A, k); it; ++it) {
				std::cout << "(" << it.row() << "," << it.col() << ") " << it.value() << std::endl;
			}
		}
	} else if (row == -1) {
		for (typename Eigen::SparseMatrix<typename Derived::Scalar>::InnerIterator it(A, col); it; ++it) {
			std::cout << "(" << it.row() << "," << it.col() << ") " << it.value() << std::endl;
		}
	} else if (col == -1) {
		for (int k=0; k < A.outerSize(); ++k) {
			for (typename Eigen::SparseMatrix<typename Derived::Scalar>::InnerIterator it(A, k); it; ++it) {
				if (it.row() == row) std::cout << "(" << it.row() << "," << it.col() << ") " << it.value() << std::endl;
			}
		}
	} else {
		for (int k=0; k < A.outerSize(); ++k) {
			for (typename Eigen::SparseMatrix<typename Derived::Scalar>::InnerIterator it(A, k); it; ++it) {
				if (it.row() == row || it.col() == col) std::cout << "(" << it.row() << "," << it.col() << ") " << it.value() << std::endl;
			}
		}
	}

}

}

/// 
/// linear algebraic extensions
/// 
namespace gp::eig {
// pairwise squared norm 
// @param A MxK(axis=0), KxM(axis=1) matrix
// @param B NxK(axis=0), KxN(axis=1) matrix
template <typename Derived1, typename Derived2, typename T = typename Derived1::Scalar>
T pairwiseSquaredNorm(const Eigen::MatrixBase<Derived1>& A,
	const Eigen::MatrixBase<Derived2>& B,
	const int axis = 0) {
	T sum = 0;
	if (axis == 0) {
		assert(A.cols() == B.cols());
		int n = static_cast<int>(A.rows());
		#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < n; i++) {
			sum += ((B.rowwise() - A.row(i)).squaredNorm());
		}

	} else if (axis == 1) {
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

// compute pseudo-inverse and multiply rhs matrix. (svd)
template <typename Derived1, typename Derived2, typename Scalar = typename Derived2::Scalar,
	int Rows = Derived2::RowsAtCompileTime, int Cols = Derived2::ColsAtCompileTime>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived2::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	pinvSolve(const Eigen::MatrixBase<Derived1>& mat, const Eigen::MatrixBase<Derived2>& rhs) {
	return mat.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
}

// compute pseudo-inverse and multiply rhs matrix. (fast)
template <typename Derived1, typename Derived2, typename Derived3, typename Scalar = typename Derived3::Scalar,
	int Rows = Derived3::RowsAtCompileTime, int Cols = Derived3::ColsAtCompileTime>
	Eigen::Matrix<Scalar, Rows, Cols, (Derived3::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)>
	pinvSolve(const Eigen::MatrixBase<Derived1>& mat, const Eigen::MatrixBase<Derived2>& diag,
	const Eigen::MatrixBase<Derived3>& rhs) {
	return mat.adjoint() * diag.asDiagonal() * rhs;
}

// convert real X2 mat to complex vector
template <typename Derived, typename Scalar = Derived::Scalar, int Rows=Derived::RowsAtCompileTime>
Eigen::Vector<std::complex<Scalar>, Rows> cmplxCast(const Eigen::MatrixBase<Derived>& mat) {
	Eigen::Vector<std::complex<Scalar>, Rows> vec(mat.rows());
	for (Eigen::Index i = 0; i < vec.rows(); i++) {
		vec[i] = std::complex<Scalar>(mat(i, 0), mat(i, 1));
	}
	return vec;
}

// convert complex vector to real X2 mat
template <typename Derived, typename Scalar = Derived::Scalar::value_type, int Rows=Derived::RowsAtCompileTime>
Eigen::Matrix<Scalar, Rows, 2> realCast(const Eigen::MatrixBase<Derived>& vec) {
	Eigen::Matrix<Scalar, Rows, 2> mat(vec.rows(), 2);
	for (Eigen::Index i = 0; i < mat.rows(); i++) {
		mat.row(i) << vec(i).real(), vec(i).imag();
	}
	return mat;
}

inline void showMatrix(const Eigen::MatrixXd& mat) {
	auto gen = std::mt19937{ std::random_device{}() };
	auto tenDigitDist = std::uniform_int_distribution<uint64_t>(1'000'000'000ull, 9'999'999'999ull);

	std::string matFname = "show_matrix_" + std::to_string(tenDigitDist(gen)) + ".temp";
	std::string pyFname = "show_matrix_" + std::to_string(tenDigitDist(gen)) + ".py";
	std::ofstream temp(matFname);
	for (Eigen::Index i = 0; i < mat.rows(); i++) {
		for (Eigen::Index t = 0; t < mat.cols() - 1; t++) {
			temp << mat(i, t) << " ";
		}
		temp << mat(i, mat.cols() - 1) << "\n";
	}
	temp.close();

	temp.open(pyFname);
	const char pycode[] =
		"# -*-coding: utf-8 -*-\n"
		"import sys\n"
		"import numpy as np\n"
		"import matplotlib.pyplot as plt\n"
		"from mpl_toolkits.axes_grid1 import make_axes_locatable\n"
		"def close_event() :\n"
		"  plt.close()\n"
		"if __name__ == '__main__':\n"
		"  fig = plt.figure()\n"
		"  file = open(sys.argv[1], 'r')\n"
		"  mat =[]\n"
		"  k1 = int(sys.argv[2])\n"
		"  k2 = int(sys.argv[3])\n"
		"  for _ in range(k1) :\n"
		"    line = list(map(float, file.readline().split(' ')))\n"
		"    if (not line) : break\n"
		"    mat.append(line)\n"
		"  np_mat = np.array(mat)\n"
		"  x = range(k2)\n"
		"  y = range(k1)\n"
		"  X, Y = np.meshgrid(x, y)\n"
		"  plt.pcolormesh(X, Y, np_mat, shading='auto')\n"
		"  plt.axis('image')\n"
		"  plt.colorbar()\n"
		"  fig, ax = plt.subplots()\n"
		"  image = ax.pcolormesh(X, Y, np_mat, shading='auto')\n"
		"  ax.axis('image')\n"
		"  divider = make_axes_locatable(ax)\n"
		"  ax_cb = divider.new_horizontal(size='2%', pad=0.05)\n"
		"  fig.add_axes(ax_cb)\n"
		"  plt.close()\n"
		"  plt.colorbar(image, cax=ax_cb)\n"
		"  plt.show()\n";
	temp << pycode;
	temp.close();

	system(("python " + pyFname + " " + matFname + " " + std::to_string(mat.rows()) + " " + std::to_string(mat.cols())).c_str());
	std::filesystem::remove(matFname);
	std::filesystem::remove(pyFname);
}
}