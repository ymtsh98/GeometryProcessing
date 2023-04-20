#pragma once

#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/// 
/// basic utility
/// 
namespace kika::eig {
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
	Eigen::Matrix<F, Eigen::Dynamic, Dim> eig(vec.size(), Dim == Eigen::Dynamic ? vec[0].size() : Dim);
	for (size_t i = 0; i < vec.size(); i++) {
		for (int j = 0; j < eig.cols(); j++) {
			eig(i, j) = static_cast<F>(vec[i][j]);
		}
	}
	return eig;
}

template <class Vector>
Eigen::Index argmax(const Vector& vec) {
	Eigen::Index max = 0;
	vec.maxCoeff(&max);
	return max;
}

template <class Vector>
Eigen::Index argmin(const Vector& vec) {
	Eigen::Index min = 0;
	vec.minCoeff(&min);
	return min;
}
}

///
/// computation utility
/// 
namespace kika::eig {

inline double cross2(const Eigen::Vector2d& u, const Eigen::Vector2d& v) {
	return u[0] * v[1] - u[1] * v[0];
}

}

/// 
/// generete utility matrix
/// 
namespace kika::eig {

template <typename T = double, typename I = size_t>
Eigen::SparseMatrix<T> circuitMat(const I n, const double self = -1.0, const double next = 1.0) {
	if (n < 50) {
		Eigen::MatrixX<T> mat = Eigen::MatrixX<T>::Zero(n, n);

		for (int i = 0, k = (int)n; i < k; i++) {
			mat(i, i) = static_cast<T>(self);
			mat(i, (i+1) % k) = static_cast<T>(next);
		}

		return mat.sparseView();
	} else {
		Eigen::SparseMatrix<T> mat(n, n);
		std::vector<Eigen::Triplet<T>> triplets;
		triplets.reserve(2 * n);

		for (int i = 0, k = (int)n; i < k; i++) {
			triplets.emplace_back(i, i, static_cast<T>(self));
			triplets.emplace_back(i, (i+1) % k, static_cast<T>(next));
		}
		mat.setFromTriplets(triplets.begin(), triplets.end());

		return mat;
	}
}

// next col : std::rotate(v.begin(), v.end() - 1, v.end());
template <typename I = size_t>
Eigen::VectorXd invCircuitMat(const I n) {
	Eigen::VectorXd vec = Eigen::VectorXd::Zero(n);

	double value = 0.5 * (n - 1) / n;
	vec[0] = -value;
	vec[1] = value;

	if (n%2) {
		for (int j = 0, k = ((int)n-1)/2-1; k > 0; j++, k--) {
			value = (double)k / n;
			vec[2+j] = value;
			vec[n-j-1] = -value;
		}
	} else {
		for (int j = 0, k = (int)n-3; k > 0; j++, k -= 2) {
			value = 0.5 * k / n;
			vec[2+j] = value;
			vec[n-j-1] = -value;
		}
	}

	return vec;
}


// a.k.a cross product matrix
// e.cross(n) == e * skewSymMat(n).T
template <typename T = double>
Eigen::Matrix3<T> skewSymMat(const Eigen::Vector3<T>& v) {
	Eigen::Matrix3<T> mat;
	mat << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
	return mat;
}

template <typename Matrix, typename Scalar = Matrix::Scalar, typename T1 = Scalar, typename T2 = Scalar>
Matrix randomMat(
	const int rows, const int cols,
	const T1& lower, const T2& higher, const unsigned int seed = std::random_device{}()) {
	std::mt19937 engine(seed);
	
	std::conditional_t<std::numeric_limits<Scalar>::is_integer,
		std::uniform_int_distribution<Scalar>, std::uniform_real_distribution<Scalar>> 
		dis(static_cast<Scalar>(lower), static_cast<Scalar>(higher));
	return Matrix::NullaryExpr(rows, cols, [&]() {return dis(engine); });
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
	pinvSolve(const Eigen::MatrixBase<Derived1>& mat, const Eigen::MatrixBase<Derived2>& diagVector,
	const Eigen::MatrixBase<Derived3>& rhs) {
	return mat.adjoint() * diagVector.asDiagonal() * rhs;
}

}

/// 
/// sparse matrix extensions
/// 
namespace kika::eig {
template <typename T = double>
Eigen::SparseMatrix<T> speye(const size_t rows, const size_t cols) {
	Eigen::SparseMatrix<T> eye(rows, cols);
	eye.setIdentity();
	return eye;
}

template <typename T = double>
Eigen::SparseMatrix<T> speye(const size_t size) {
	return speye<T>(size, size);
}

/// random sparse matrix (0.0 - 1.0)
template <typename T=double>
Eigen::SparseMatrix<T> sprand(const size_t rows, const size_t cols, const double density, const unsigned int seed = std::random_device{}()) {
	std::mt19937 engine(seed);

	std::uniform_real_distribution<T> dist(0.0, 1.0);

	std::vector<Eigen::Triplet<T>> triplets;
	triplets.reserve((size_t)(rows * cols * density));
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

// vector as diagonal sparse matrix.
template <typename T = double>
Eigen::SparseMatrix<T> spdiag(const Eigen::VectorX<T>& diag) {
	Eigen::SparseMatrix<T> mat(diag.rows(), diag.rows());
	mat.setIdentity();
	mat.diagonal().array() *= diag.array();
	return mat;
}

/// @param offset : start index offset
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

/// @param offset : start index offset
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

// horizontally (row_A, col_A + col_B) : Sparse + Sparse
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

// vertically (row_A + row_B, col_A) : Sparse + Sparse
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

// vertically (row_A + row_B, col_A) : Sparse + Diagnoal
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

// print sparse matrix non-zero elements.
// warnings : row >= 0 && col >= 0 is "or" operation. print elements under row "_" col.
// "or"  print(spmat, row, col)
// "and" print(spmat.topLeftCorner(row, col).eval()).
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
/// dense matrix extensions
/// 
namespace kika::eig {

// normalize 0 - 1 range
template <typename Derived, typename Scalar = Derived::Scalar, 
	int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime>
inline Eigen::Matrix<Scalar, Rows, Cols> zero2oneNorm(const Eigen::DenseBase<Derived>& vector) {
	Eigen::Matrix<Scalar, Rows, Cols> ret = vector;
	ret.array() -= ret.minCoeff();
	ret.array() /= ret.maxCoeff();
	return ret;
}

// append value
template <typename Scalar>
void append(Eigen::Vector<Scalar, -1>& vector, const Scalar value) {
	vector.conservativeResize(vector.rows() + 1);
	vector(vector.rows() - 1) = value;
}

#if false
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
#endif

namespace impl {
template <typename Derived, typename... Args>
void concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(const Eigen::DenseBase<Derived>& A, const Args&... args) {
	if constexpr (sizeof...(args) != 0) concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);
}

template <typename Derived1, typename Derived2, typename... Args>
void hAppendDenseMatImpl(Eigen::PlainObjectBase<Derived1>& A, const Eigen::Index& offset, const Eigen::DenseBase<Derived2>& B, const Args&... args) {
	A.block(0, offset, A.rows(), B.cols()) = B;
	if constexpr (sizeof...(args) != 0) hAppendDenseMatImpl(A, offset + B.cols(), args...);
}

template <typename Derived1, typename Derived2, typename... Args>
void vAppendDenseMatImpl(Eigen::PlainObjectBase<Derived1>& A, const Eigen::Index& offset, const Eigen::DenseBase<Derived2>& B, const Args&... args) {
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
template <typename Derived, typename... Args>
void hAppendDenseMat(Eigen::PlainObjectBase<Derived>& A, const Args&... args) {
	assert(((A.rows() == args.rows()) && ...));
	impl::concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);

	Eigen::Index offset = A.cols();
	A.conservativeResize(Eigen::NoChange, A.cols() + (args.cols() + ...));

	impl::hAppendDenseMatImpl(A, offset, args...);
}

//vertically (row_A + row_B, col_A) : Dense + Dense
template <typename Derived, typename... Args>
void vAppendDenseMat(Eigen::PlainObjectBase<Derived>& A, const Args&... args) {
	assert(((A.cols() == args.cols()) && ...));
	impl::concatNappendFunctionCanOnlyBePassedDenseBaseMatrix(args...);

	Eigen::Index offset = A.rows();
	A.conservativeResize(A.rows() + (args.rows() + ...), Eigen::NoChange);

	impl::vAppendDenseMatImpl(A, offset, args...);
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

}

/// 
/// operators
/// 
namespace kika::eig {

// this is same operator (circutMat * mat) / (mat * circuitMat)
// usage : 
// circuitMat(n, -1, 1)   :(CircuitOperator - mat) / (mat - CircuitOperator)
// circuitMat(n, 1, 1)    :(CircuitOperator + mat) / (mat + CircuitOperator)
// circuitMat(n, -1, 1).transpose() :(CircuitOperator * mat) / (mat * CircuitOperator)
class CircuitOperator {
	std::vector<size_t> forward, backward;

	template <typename T>
	void resizeForward(const T& size) {
		if (forward.size() != (size_t)size) {
			forward.resize(size);
			std::iota(forward.begin(), forward.end(), 1);
			forward.back() = 0;
		}
	}

	template <typename T>
	void resizeBackward(const T& size) {
		if (backward.size() != (size_t)size) {
			backward.resize(size);
			backward[0] = size - 1;
			std::iota(backward.begin() + 1, backward.end(), 0);
		}
	}

public:
	CircuitOperator(const size_t reserve = 3) {
		forward.reserve(reserve);
		backward.reserve(reserve);
	}

	template <typename Matrix>
	auto operator+(const Matrix& mat) {
		resizeForward(mat.rows());
		return mat + mat(forward, Eigen::all);
	}

	template <typename Matrix>
	friend auto operator+(const Matrix& mat, CircuitOperator& op) {
		op.resizeBackward(mat.cols());
		return mat + mat(Eigen::all, op.backward);
	}

	template <typename Matrix>
	auto operator-(const Matrix& mat) {
		resizeForward(mat.rows());
		return mat(forward, Eigen::all) - mat;
	}

	template <typename Matrix>
	friend auto operator-(const Matrix& mat, CircuitOperator& op) {
		op.resizeBackward(mat.cols());
		return mat(Eigen::all, op.backward) - mat;
	}

	template <typename Matrix>
	auto operator*(const Matrix& mat) {
		resizeBackward(mat.rows());
		return mat(backward, Eigen::all) - mat;
	}

	template <typename Matrix>
	friend auto operator*(const Matrix& mat, CircuitOperator& op) {
		op.resizeForward(mat.cols());
		return mat(Eigen::all, op.forward) - mat;
	}
};

}