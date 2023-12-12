#pragma once

#include <set>
#include <kika/sampling.hpp>
#pragma warning (push) 
#pragma warning (disable : 4244)
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
#pragma warning(pop)

namespace kika::util {

namespace impl {
template <typename Scalar_, int Uplo = Eigen::Lower, int Flags = Eigen::ColMajor, typename StorageIndex = int>
class SparseSymShiftSolveChol {
public:
  using Scalar = Scalar_;

private:
  using Index = Eigen::Index;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MapConstVec = Eigen::Map<const Vector>;
  using MapVec = Eigen::Map<Vector>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar, Flags, StorageIndex>;
  using ConstGenericSparseMatrix = const Eigen::Ref<const SparseMatrix>;

  ConstGenericSparseMatrix m_mat;
  const Index m_n;
  Eigen::SimplicialLDLT<SparseMatrix> m_solver;

public:
  template <typename Derived>
	SparseSymShiftSolveChol(const Eigen::SparseMatrixBase<Derived>& mat) :
    m_mat(mat), m_n(mat.rows()) {
    static_assert(
      static_cast<int>(Derived::PlainObject::IsRowMajor) == static_cast<int>(SparseMatrix::IsRowMajor),
      "SparseSymShiftSolveChol: the \"Flags\" template parameter does not match the input matrix (Eigen::ColMajor/Eigen::RowMajor)");

    if (mat.rows() != mat.cols())
      throw std::invalid_argument("SparseSymShiftSolveChol: matrix must be square");
  }

  Index rows() const { return m_n; }
  Index cols() const { return m_n; }

  void set_shift(const Scalar& sigma) {
    SparseMatrix mat = m_mat.template selfadjointView<Uplo>();
    SparseMatrix identity(m_n, m_n);
    identity.setIdentity();
    mat = mat - sigma * identity;

    m_solver.compute(mat);
    if (m_solver.info() != Eigen::Success)
      throw std::invalid_argument("SparseSymShiftSolveChol: factorization failed with the given shift");
  }

  void perform_op(const Scalar* x_in, Scalar* y_out) const {
    MapConstVec x(x_in, m_n);
    MapVec y(y_out, m_n);
    y.noalias() = m_solver.solve(x);
  }
};

}

class LboEigs {
	kika::heds::SurfaceMesh& mesh;
	kika::diff::DiscreteLaplacian& dsl;

private:

	// main impl
	// 
	// void eigCpu(const Eigen::SparseMatrix<double>& L, const Eigen::SparseMatrix<double>& M, const size_t neigs) {
	// 	try {
	// 		impl::SparseSymShiftSolveChol<double> op(L);
	// 		Spectra::SparseSymMatProd<double> bop(M);
	// 		Spectra::SymGEigsShiftSolver<decltype(op), decltype(bop), Spectra::GEigsMode::ShiftInvert> eigs(op, bop, neigs, 2 * neigs, 1e-8);
	// 		eigs.init();
	// 		eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10, Spectra::SortRule::SmallestMagn);
	// 		evals = eigs.eigenvalues();
	// 		evecs = eigs.eigenvectors();
	// 
	// 	} catch (std::exception&) {
	// 		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(L, M, Eigen::ComputeEigenvectors);
	// 		evals = solver.eigenvalues();
	// 		evecs = solver.eigenvectors();
	// 	}
	// }

	// sometime Spectra failed to compute eig
	// we try fast -> normal -> slow
	void eigCpu(const Eigen::SparseMatrix<double>& L, const Eigen::SparseMatrix<double>& M, const size_t neigs, 
		const Eigen::SparseMatrix<double>& origL, const Eigen::VectorXd& origM) {
		try {
			impl::SparseSymShiftSolveChol<double> op(L);
			Spectra::SparseSymMatProd<double> bop(M);
			Spectra::SymGEigsShiftSolver<decltype(op), decltype(bop), Spectra::GEigsMode::ShiftInvert> eigs(op, bop, neigs, 2 * neigs, 1e-8);
			eigs.init();
			eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10, Spectra::SortRule::SmallestMagn);
			evals = eigs.eigenvalues();
			evecs = eigs.eigenvectors();

		} catch (std::exception&) {
			
			try {
				Eigen::VectorXd u = dsl.s0.cwiseSqrt().cwiseInverse();
				Eigen::SparseMatrix<double> L = u.asDiagonal() * dsl.lbo * u.asDiagonal();
				impl::SparseSymShiftSolveChol<double> op(L);
				Spectra::SymEigsShiftSolver<impl::SparseSymShiftSolveChol<double>> eigs(op, neigs, 2 * neigs, 1e-8);
				eigs.init();
				eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10, Spectra::SortRule::SmallestMagn);
				evals = eigs.eigenvalues().cwiseAbs();
				evecs = u.asDiagonal() * eigs.eigenvectors();

			} catch (std::exception&) {
				Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(L, M, Eigen::ComputeEigenvectors);
				evals = solver.eigenvalues();
				evecs = solver.eigenvectors();
			}

		}
	}

public:
	Eigen::VectorXd evals;
	Eigen::MatrixXd evecs;

	// @param sampleSize : (defalut) 5 * n
	LboEigs(kika::heds::SurfaceMesh& mesh, kika::diff::DiscreteLaplacian& dsl) :
		mesh(mesh), dsl(dsl) {

		dsl.laplacianOp();
		dsl.vertexAreas();
	}

	template <bool Prolonged = true>
	std::pair<std::vector<uint32_t>, Eigen::SparseMatrix<double>> compute(const size_t n, size_t sampleSize = 0) {
		if (sampleSize == 0) sampleSize = 5 * n;
		if (sampleSize < n) sampleSize = n;
		if (sampleSize > mesh.getNumVerts()) sampleSize = mesh.getNumVerts();

		kika::samp::BlueNoiseSampler sampler(mesh);
		sampler.sampling(sampleSize);
		auto& prolongOp = sampler.prolong();

		Eigen::SparseMatrix<double> L = prolongOp.transpose() * dsl.lbo * prolongOp;
		Eigen::SparseMatrix<double> M = prolongOp.transpose() * dsl.s0.asDiagonal() * prolongOp;
		eigCpu(L, M, n, dsl.lbo, dsl.s0);

		Eigen::VectorXd m = (evecs.transpose() * M * evecs).diagonal();
		evecs = evecs.array().rowwise() * m.cwiseSqrt().cwiseInverse().array().transpose();
		if constexpr (Prolonged) evecs = (prolongOp * evecs).leftCols(n).eval();
		else evecs = evecs.leftCols(n).eval();
		evals = evals.cwiseAbs().topRows(n).eval();

		return std::make_pair(sampler.samples, sampler.prolongOp);
	}
	 
	template <bool Prolonged = true>
	void compute(const size_t n, const kika::samp::BlueNoiseSampler& sampler) {
		auto& prolongOp = sampler.prolongOp;

		Eigen::SparseMatrix<double> L = prolongOp.transpose() * dsl.lbo * prolongOp;
		Eigen::SparseMatrix<double> M = prolongOp.transpose() * dsl.s0.asDiagonal() * prolongOp;
		eigCpu(L, M, n, dsl.lbo, dsl.s0);

		Eigen::VectorXd m = (evecs.transpose() * M * evecs).diagonal();
		evecs.array().rowwise() *= m.cwiseSqrt().cwiseInverse().array().transpose();
		if constexpr (Prolonged) evecs = (prolongOp * evecs).leftCols(n).eval();
		else evecs = evecs.leftCols(n).eval();
		evals = evals.cwiseAbs().topRows(n).eval();
	}

};

}