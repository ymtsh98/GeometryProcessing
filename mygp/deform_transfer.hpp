#pragma once
namespace gp {
/*
Deformation Transfer for Triangle Meshes
Robert W.Sumner and Javan PopovicÅL
*/
class TriDeformTransfer {

	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd affineMatrices;
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	size_t ns;

	void prepareLhs(const Eigen::MatrixXd& originalSrc, const Eigen::MatrixXi& srcFaces) {
		Eigen::Index nv = originalSrc.rows(), nf = srcFaces.rows();
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(9*nf);

		for (Eigen::Index i = 0; i < nf; i++) {
			Eigen::Vector3i face = srcFaces.row(i);
			Eigen::Matrix3d fix;
			fix.row(0) = originalSrc.row(face[1]) - originalSrc.row(face[0]);
			fix.row(1) = originalSrc.row(face[2]) - originalSrc.row(face[0]);
			fix.row(2) = fix.row(0).cross(fix.row(1)).normalized();
			fix = fix.inverse().eval();

			for (size_t j = 0, r = 0; j < 3; j++) {
				for (size_t k = 0; k < 3; k++, r++) {
					int row = static_cast<int>(3 * 3 * i + r);
					triplets.emplace_back(row, static_cast<int>(3 * face[0] + j), -fix(k, 0));
					triplets.emplace_back(row, static_cast<int>(3 * face[0] + j), -fix(k, 1));
					triplets.emplace_back(row, static_cast<int>(3 * face[1] + j), fix(k, 0));
					triplets.emplace_back(row, static_cast<int>(3 * face[2] + j), fix(k, 1));
					triplets.emplace_back(row, static_cast<int>(3 * (nv + i) + j), fix(k, 2));
				}
			}
		}

		A.resize(3 * 3 * nf, 3 * (nv + nf));
		A.setFromTriplets(triplets.begin(), triplets.end());

		solver.compute(A.transpose() * A);
	}

	void prepareAffineMatrices(const Eigen::MatrixXd& originalTrg, const Eigen::MatrixXd& deformedTrg, const Eigen::MatrixXi& trgFaces) {
		Eigen::Index rows = trgFaces.rows();
		affineMatrices.resize(3 * 3 * rows);

		for (Eigen::Index i = 0; i < rows; i++) {
			Eigen::Matrix3d org, def;
			org.row(0) = originalTrg.row(trgFaces(i, 1)) - originalTrg.row(trgFaces(i, 0));
			org.row(1) = originalTrg.row(trgFaces(i, 2)) - originalTrg.row(trgFaces(i, 0));
			org.row(2) = org.row(0).cross(org.row(1)).normalized();
			def.row(0) = deformedTrg.row(trgFaces(i, 1)) - deformedTrg.row(trgFaces(i, 0));
			def.row(1) = deformedTrg.row(trgFaces(i, 2)) - deformedTrg.row(trgFaces(i, 0));
			def.row(2) = def.row(0).cross(def.row(1)).normalized();

			Eigen::Matrix3d affine = org.inverse() * def;
			affineMatrices.block(i * 9, 0, 9, 1) = affine.reshaped(9, 1);
		}
	}

public:


	TriDeformTransfer(const Eigen::MatrixXd& originalSrc, const Eigen::MatrixXi& srcFaces,
		const Eigen::MatrixXd& originalTrg, const Eigen::MatrixXd& deformedTrg, const Eigen::MatrixXi& trgFaces) :
		ns(static_cast<size_t>(originalSrc.rows())) {
		prepareLhs(originalSrc, srcFaces);
		prepareAffineMatrices(originalTrg, deformedTrg, trgFaces);
	}

	template <typename T>
	Eigen::MatrixXd run(const std::vector<T>& corres) {
		size_t nf = corres.size();

		Eigen::VectorXd rhs(3 * 3 * nf);
		for (size_t i = 0; i < nf; i++) {
			rhs.block(i * 9, 0, 9, 1) = affineMatrices.block(corres[i] * 9, 0, 9, 1);
		}

		return solver.solve(A.transpose() * rhs).topRows(ns * 3).reshaped(3, ns).transpose();
	}
};
}