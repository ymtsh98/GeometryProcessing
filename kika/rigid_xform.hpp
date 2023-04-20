#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

namespace kika::util {

/// rigid rotation & translation solver
/// trg = rot * src + xlate;
template <
	typename DerivedSrc, typename DerivedTrg,
	int SrcRows = DerivedSrc::RowsAtCompileTime, int SrcCols = DerivedSrc::ColsAtCompileTime,
	int TrgRows = DerivedTrg::RowsAtCompileTime, int TrgCols = DerivedTrg::ColsAtCompileTime,
	typename Scalar = DerivedSrc::Scalar>
class RigidXformSolver {
	Eigen::JacobiSVD<Eigen::Matrix<Scalar, SrcCols, TrgCols>> solver;
	Eigen::Matrix<Scalar, SrcRows, SrcCols> src;
	Eigen::Matrix<Scalar, TrgRows, TrgCols> trg;
	Eigen::Vector<Scalar, SrcCols> srcCenter;
	Eigen::Vector<Scalar, TrgCols> trgCenter;

public:
	Eigen::Matrix<Scalar, SrcCols, TrgCols> rot;
	Eigen::Vector<Scalar, SrcCols> xlate;

private:
	void compute() {
		srcCenter = src.colwise().sum() / src.rows();
		trgCenter = trg.colwise().sum() / trg.rows();
		Eigen::Matrix<Scalar, SrcCols,TrgRows> && S = (src.rowwise() - srcCenter.transpose()).transpose()* (trg.rowwise() - trgCenter.transpose());
		if constexpr (SrcCols == -1 || TrgCols == -1) solver.compute(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
		else solver.compute(S, Eigen::ComputeFullU | Eigen::ComputeFullV);

		auto V = solver.matrixV();
		auto U = solver.matrixU();

		Eigen::Vector<Scalar, SrcCols> reflect = Eigen::Vector<Scalar, SrcCols>::Ones(S.rows());
		reflect(S.rows()-1) = (V * U.transpose()).determinant();

		rot = V * reflect.asDiagonal() * U.transpose();
		xlate = trgCenter - rot * srcCenter;
	}

public:

	/// rigid rotation & translation solver
	/// trg.row(i) = rot * src.row(i).T + xlate;
	RigidXformSolver() {}

	/// rigid rotation & translation solver
	/// trg.row(i) = rot * src.row(i).T + xlate;
	/// @param src MxK1
	/// @param trg NxK2
	RigidXformSolver(const Eigen::DenseBase<DerivedSrc>& src, const Eigen::DenseBase<DerivedTrg>& trg) : src(src), trg(trg) {
		compute();
	}

	void solve(const Eigen::Matrix<Scalar, SrcRows, SrcCols>& src, const Eigen::Matrix<Scalar, TrgRows, TrgCols>& trg) {
		this->src = src;
		this->trg = trg;
		compute();
	}
};
}