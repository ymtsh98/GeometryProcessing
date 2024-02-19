#pragma once

namespace kika::hdif {

class LaplacianSmoother {
	Eigen::SparseMatrix<double> smoother;

public:

	LaplacianSmoother(const Eigen::SparseMatrix<double>& lap, const Eigen::VectorXd& vareas) {
		Eigen::VectorXd invA = vareas.cwiseInverse();
		smoother = invA.asDiagonal() * lap;
		invA = smoother.diagonal();
		invA = invA.asDiagonal().inverse().diagonal();
		smoother = invA.asDiagonal() * smoother;
	}

	template <typename Derived, typename Scalar = typename Derived::Scalar,
		int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime, bool IsRowMajor = Derived::IsRowMajor>
	Eigen::Matrix<Scalar, Rows, Cols, IsRowMajor> run(const Eigen::DenseBase<Derived>& values, const size_t iter = 1, const double shrink = 0.1, const double inflate = 0.0) {
		Eigen::Matrix<Scalar, Rows, Cols, IsRowMajor> ret = values;

		for (size_t i = 0; i < iter; i++) {
			if (inflate > 0.) ret += inflate * smoother * ret;
			if (shrink > 0.) ret -= shrink * smoother * ret;
		}

		return ret;
	}
};

class LaplacianInpainter {
	std::vector<uint32_t> paint, inpaint;
	Eigen::SparseMatrix<double> p2ip; // painted value to inpainted
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; // dirchlet + laplacian solver

public:

	template <typename T = int>
	LaplacianInpainter(const Eigen::SparseMatrix<double>& lap, const Eigen::VectorXd& vareas,
		const std::vector<T>& inpaintTarget) {
		Eigen::SparseMatrix<double> lhs = -lap + lap * vareas.cwiseInverse().asDiagonal() * lap;

		inpaint = kika::util::vecast<uint32_t>(inpaintTarget);
		paint = kika::util::complement((size_t)vareas.rows(), inpaint);

		solver.compute(kika::eig::spsub(lhs, inpaint));
		p2ip = kika::eig::spsub(lhs, inpaint, paint);
	}

	template <typename Derived, typename Scalar = typename Derived::Scalar,
		int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime, bool IsRowMajor = Derived::IsRowMajor>
	Eigen::Matrix<Scalar, Rows, Cols, IsRowMajor> run(const Eigen::DenseBase<Derived>& values) {
		Eigen::Matrix<Scalar, Rows, Cols, IsRowMajor> ret = values;
		ret(inpaint, Eigen::all) = solver.solve(-p2ip * values(paint, Eigen::all)).eval();
		return ret;
	}
};

}