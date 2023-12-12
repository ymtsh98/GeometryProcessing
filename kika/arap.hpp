#pragma once
#include <kika/kikautil.hpp>
#include <kika/eigutil.hpp>
#include <set>

namespace kika::def {

// #define LOCAL_ARAP_RESTRICT_DEBUG 

//// ARAP shape deformer
//// Local Deformation for Interactive Shape Editing (2023)
template <bool SpokesAndRims = false>
class LocalARAPDeformer {
	std::vector<std::vector<uint32_t>> onerings; // counter-clockwise order onering neighbs
	std::vector<Eigen::VectorXd> wspokes, wrims; // weight of spokes, rims
	std::vector<Eigen::Matrix3d> rot; // vertex-wise rotation matrices (restricted)
	Eigen::VectorXd vareas; // vertex areas (restricted)

	double w, s; // region of influence
	double r, eps = 1e-6; // r = f(w, s, vareas) + eps;
	double sre; // smooth rotation enhance coeff

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

	std::vector<uint32_t> computeIdx; // compute region index
	std::vector<uint32_t> boundaryIdx; // index of between compute and noncompute
	std::vector<uint32_t> restrictIdx; // computeIdx + restrictIdx
	std::vector<uint32_t> full2comp; // full mesh index to compute region index
	std::vector<bool> uncompute; // is i-th vertex computed

	bool classicARAP;

public:

	LocalARAPDeformer() : w(1.), s(1.), r(1.), sre(0.04), classicARAP(false) {}

	// @param computeVertIdx : (optional) compute region
	// @param controlVertIdx : (optional) hard constraint control region
	// ((~computeVertIdx) + controlVertIdx) will be fixed)
	template <typename T1 = int, typename T2 = int>
	LocalARAPDeformer(const kika::heds::SurfaceMesh& mesh, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
		const double w = 1.0, const double s = 0.001, const std::vector<T1>& computeVertIdx ={}, const std::vector<T2>& controlVertIdx ={})
		: w(1.), s(1.), r(1.), sre(0.04), classicARAP(false) {

		build(mesh, lbo, vareas, w, s, computeVertIdx, controlVertIdx);
	}

	// build deformer
	// @param computeVertIdx : (optional) compute region
	// @param controlVertIdx : (optional) hard constraint control region
	// ((~computeVertIdx) + controlVertIdx) will be fixed)
	template <typename T1 = int, typename T2 = int>
	void build(const kika::heds::SurfaceMesh& mesh, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
		const double w = 1.0, const double s = 0.001, const std::vector<T1>& computeVertIdx ={}, const std::vector<T2>& controlVertIdx ={}) {
		updateWeights(mesh, lbo);
		updateComputeMask(computeVertIdx, controlVertIdx);
		updateLinearSolver(lbo, vareas, w, s);
	}

	// collect laplacian coeff
	void updateWeights(const kika::heds::SurfaceMesh& mesh, const Eigen::SparseMatrix<double>& lbo) {
		size_t nv = mesh.getNumVerts();
		onerings.resize(nv);
		wspokes.resize(nv);
		if constexpr (SpokesAndRims) wrims.resize(nv);

		for (size_t i = 0; i < nv; i++) {
			onerings[i] = mesh.verts[i]->oneringNeighbs<kika::heds::Idx, kika::heds::CCW>();
			wspokes[i].resize(onerings[i].size());
			if constexpr (SpokesAndRims) wrims[i].resize(onerings[i].size());

			for (size_t j = 0, k = onerings[i].size(); j < k; j++) {
				wspokes[i][j] = lbo.coeff(onerings[i][j], i);
				if constexpr (SpokesAndRims) wrims[i][j] = lbo.coeff(onerings[i][j], onerings[i][(j + 1) % k]);
			}
		}
	}

	// update compute mask
	template <typename T1 = int, typename T2 = int>
	void updateComputeMask(const std::vector<T1>& computeVertIdx ={}, const std::vector<T2>& controlVertIdx ={}) {
		size_t nv = onerings.size();

		if (computeVertIdx.size() == 0) {
			computeIdx.resize(nv);
			std::iota(computeIdx.begin(), computeIdx.end(), 0);
		} else {
			computeIdx = kika::util::vecast<uint32_t>(computeVertIdx);
		}

		uncompute = std::vector<bool>(nv, true);
		for (auto i : computeIdx) uncompute[i] = false;
		for (auto i : controlVertIdx) uncompute[i] = true;

		computeIdx.clear();
		boundaryIdx.clear();
		restrictIdx.clear();
		full2comp = std::vector<uint32_t>(nv, std::numeric_limits<uint32_t>::max());
		std::vector<bool> added(nv, false);

		for (size_t i = 0; i < nv; i++) {
			if (!uncompute[i]) {
				full2comp[i] = (uint32_t)computeIdx.size();
				computeIdx.emplace_back((uint32_t)i);
				restrictIdx.emplace_back((uint32_t)i);
				added[i] = true;
			}
		}

		// collect boundary(compute-uncompute bride)
		for (const auto i : computeIdx) {
			for (const auto j : onerings[i]) {
				if (!added[j] && uncompute[j]) {
					full2comp[j] = (uint32_t)(computeIdx.size() + boundaryIdx.size());
					boundaryIdx.emplace_back((uint32_t)j);
					restrictIdx.emplace_back((uint32_t)j);
					added[j] = true;
				}
			}
		}

	}

	// update global position solver
	void updateLinearSolver(const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas, const double w = 1.0, const double s = 0.001) {
		size_t nv = onerings.size();

		r = (w * vareas.array()).matrix().maxCoeff() / s + eps;
		if (classicARAP) r = 0.;

		if (computeIdx.size() == nv) solver.compute(lbo + r * kika::eig::speye(vareas.rows()));
		else solver.compute(kika::eig::spsub(lbo + r * kika::eig::speye(vareas.rows()), computeIdx));

		this->w = w;
		this->s = s;
		this->vareas = vareas(restrictIdx);

		#ifdef LOCAL_ARAP_RESTRICT_DEBUG
		this->vareas = vareas;
		#endif
	}

	// use classic ARAP methods
	// WARNING : can not use soft constraint in the same time
	void useClassicARAP(const bool use) {
		classicARAP = use;
	}

	// update smooth rotation enhance coeff
	void setSmoothRotEnhance(const double coeff) {
		this->sre = coeff;
	}

private:

	// precompute fixed lhs solution values (this will be added to rhs)
	Eigen::MatrixX3d computeFixedLhs(const Eigen::MatrixX3d& pos) {
		Eigen::MatrixX3d fixedLhs = Eigen::MatrixX3d::Zero(pos.rows(), 3);

		for (auto i : computeIdx) {
			for (size_t j = 0; j < onerings[i].size(); j++) {
				if (uncompute[onerings[i][j]]) {
					fixedLhs.row(i) -= wspokes[i][j] * pos.row(onerings[i][j]);
				}
			}
		}

		return fixedLhs;
	}

	// compute rotation (svd or eig)
	Eigen::Matrix3d computeRotation(const Eigen::Matrix3d& cov) {
		#if true
		Eigen::JacobiSVD<Eigen::Matrix3d> svds(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

		auto& U = svds.matrixU();
		auto& V = svds.matrixV();
		Eigen::Vector3d D{ 1., 1., (V * U.transpose()).determinant() };
		return V * D.asDiagonal() * U.transpose();

		#else
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigs;
		eigs.computeDirect(cov.transpose() * cov, Eigen::ComputeEigenvectors);

		Eigen::Vector3d s = eigs.eigenvalues().cwiseSqrt().cwiseInverse();
		return (cov * eigs.eigenvectors() * s.asDiagonal() * eigs.eigenvectors().transpose()).transpose();

		#endif
	}

	// compute per-vertex rotation matrix
	void localRotations(const Eigen::MatrixX3d& defm, const Eigen::MatrixX3d& rest) {
		Eigen::VectorXd norm = (defm - rest).rowwise().squaredNorm();
		int n = (int)norm.rows();

		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			#ifdef LOCAL_ARAP_RESTRICT_DEBUG
			int vi = i;
			auto& onering = onerings[vi];
			#else
			int vi = restrictIdx[i];
			auto onering = onerings[vi];
			for (size_t j = 0; j < onering.size(); j++) onering[j] = full2comp[onering[j]];
			#endif

			if (uncompute[vi] || (norm(i) < eps && (norm(onering).array() < eps).count() == onering.size())) {
				rot[i].setIdentity();
				continue;
			}

			Eigen::MatrixX3d dX = defm(onering, Eigen::all), rX = rest(onering, Eigen::all);

			// spokes term
			Eigen::VectorXd& spokes = wspokes[vi];
			Eigen::MatrixX3d dE = dX.array().rowwise() - defm.row(i).array();
			Eigen::MatrixX3d rE = rX.array().rowwise() - rest.row(i).array();
			Eigen::Matrix3d cov = -rE.transpose() * spokes.asDiagonal() * dE;

			// rims term
			if constexpr (SpokesAndRims) {
				Eigen::VectorXd& rims = wrims[vi];
				auto circuit = kika::eig::circuitMat(onering.size(), -1., 1.);
				dE = circuit * dX;
				rE = circuit * rX;
				cov += -rE.transpose() * rims.asDiagonal() * dE;
			}

			// smooth rotation enhance
			double coeff = sre * vareas[i];
			if (sre < 0.) {
				for (size_t j = 0; j < onering.size(); j++) cov -= coeff * spokes[j] * rot[onering[j]];
			} else if (sre > 0.) {
				for (size_t j = 0; j < onering.size(); j++) cov -= coeff * spokes[j] * rot[onering[j]].transpose();
			}

			rot[i] = computeRotation(cov);
		}
	}

	// SC-L1
	Eigen::MatrixX3d smoothlyClampedL1Norm(const Eigen::MatrixX3d& defm, const Eigen::MatrixX3d& rest, const Eigen::MatrixX3d& U) {
		Eigen::MatrixX3d Z = defm - rest + U;
		Eigen::VectorXd znorm = Z.rowwise().norm().cwiseMax(1e-20);
		Eigen::VectorXd shrink = ((r * s - w * vareas.array() * (s / znorm.array())) / (r * s - w * vareas.array())).cwiseMax(0);
		Eigen::VectorXd local = (znorm.array() > s).select(1., shrink.array());
		Z = local.asDiagonal() * Z;
		return Z;
	}

	// compute original ARAP rhs
	Eigen::MatrixX3d arapRhs(const Eigen::MatrixX3d& rest) {
		Eigen::MatrixX3d rhs = Eigen::MatrixX3d::Zero(rest.rows(), 3);
		int n = (int)rhs.rows();

		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			#ifdef LOCAL_ARAP_RESTRICT_DEBUG
			int vi = i;
			auto& onering = onerings[vi];
			#else
			int vi = restrictIdx[i];
			auto onering = onerings[vi];
			for (size_t j = 0; j < onering.size(); j++) onering[j] = full2comp[onering[j]];
			#endif

			if (uncompute[vi]) continue;

			Eigen::VectorXd& spokes = wspokes[vi];

			for (size_t j = 0, k = onering.size(); j < k; j++) {
				if constexpr (SpokesAndRims) {
					Eigen::VectorXd& rims = wrims[vi];

					double denom = 4.;
					Eigen::Matrix3d R = 2. * rot[onering[j]] + 2. * rot[i];

					// boudnary condition (this is lazy condition), just skip
					auto v = onering[(j + onering.size() - 1) % k], u = onering[(j + 1) % k];
					if (rims[(j + onering.size() - 1) % k] != 0.) { R += rot[v]; denom += 1.; }
					if (rims[(j + 1) % k] != 0.) { R += rot[u]; denom += 1.; }

					rhs.row(i) += (1. / denom) * spokes[j] * R * (rest.row(onering[j]) - rest.row(i)).transpose();
				} else {
					rhs.row(i) += 0.5 * spokes[j] * (rot[onering[j]] + rot[i]) * (rest.row(onering[j]) - rest.row(i)).transpose();
				}
			}
		}

		return rhs;
	}

public:

	// WARNINGS : position of softControlIdx will not be preserved
	// @param defm : computeIdx (+controlIdx) is seted in updateComputeMask, defm(...) will be preserved
	// @param softControlIdx : soft constraint controller vertex index
	template <typename T = int>
	Eigen::MatrixX3d run(Eigen::MatrixX3d defm, const Eigen::MatrixX3d& rest, const size_t iter, const std::vector<T>& softControlIdx ={}) {

		#ifndef LOCAL_ARAP_RESTRICT_DEBUG

		size_t rv = restrictIdx.size();
		
		rot.resize(rv);
		std::fill(rot.begin(), rot.end(), Eigen::Matrix3d::Identity());
		
		Eigen::MatrixX3d U = Eigen::MatrixX3d::Zero(rv, 3);
		
		Eigen::MatrixX3d restRestrict = rest(restrictIdx, Eigen::all);
		Eigen::MatrixX3d defmRestrict = defm(restrictIdx, Eigen::all);
		Eigen::MatrixX3d fixedLhs = computeFixedLhs(defm);
		Eigen::MatrixX3d fixedLhsRestrict = fixedLhs(restrictIdx, Eigen::all);
		
		std::vector<uint32_t> softControlIdxRestrict;
		softControlIdxRestrict.reserve(softControlIdx.size());
		for (size_t i = 0, c = 0; i < softControlIdx.size(); i++) {
			if (full2comp[softControlIdx[i]] != std::numeric_limits<uint32_t>::max()) {
				softControlIdxRestrict.emplace_back(full2comp[softControlIdx[i]]);
			}
		}
		Eigen::MatrixX3d ctrlRestrict = defmRestrict(softControlIdxRestrict, Eigen::all);

		for (size_t step = 0; step < iter; step++) {
			// soft constrain controller position
			if (softControlIdxRestrict.size()) defmRestrict(softControlIdxRestrict, Eigen::all) = ctrlRestrict;

			// step 1
			if (step != 0) localRotations(defmRestrict, restRestrict);

			// // step 2
			Eigen::MatrixX3d Z = smoothlyClampedL1Norm(defmRestrict, restRestrict, U);
			Eigen::MatrixX3d scl1Rhs = r * (restRestrict + Z - U);
			if (classicARAP) scl1Rhs.setZero();

			// step -
			Eigen::MatrixX3d B = scl1Rhs + fixedLhsRestrict + arapRhs(restRestrict);

			// step 3
			defmRestrict.topRows(computeIdx.size()) = solver.solve(B.topRows(computeIdx.size())).eval();

			// step 4
			U = U + defmRestrict - restRestrict - Z;
		}
		
		defm(computeIdx, Eigen::all) = defmRestrict.topRows(computeIdx.size());
		
		return defm;

		#else

		size_t nv = (size_t)defm.rows();

		rot.resize(nv);
		std::fill(rot.begin(), rot.end(), Eigen::Matrix3d::Identity());

		Eigen::MatrixX3d U = Eigen::MatrixX3d::Zero(nv, 3);

		Eigen::MatrixX3d fixedLhs = computeFixedLhs(defm);
		Eigen::MatrixX3d ctrl = defm(softControlIdx, Eigen::all);

		for (size_t step = 0; step < iter; step++) {
			// soft constrain controller position
			if (softControlIdx.size()) defm(softControlIdx, Eigen::all) = ctrl;

			// step 1
			if (step != 0) localRotations(defm, rest);

			// step 2
			Eigen::MatrixX3d Z = smoothlyClampedL1Norm(defm, rest, U);
			Eigen::MatrixX3d scl1Rhs = r * (rest + Z - U);
			if (classicARAP) scl1Rhs.setZero();

			// step -
			Eigen::MatrixX3d B = fixedLhs + scl1Rhs + arapRhs(rest);

			// step 3
			if (computeIdx.size() == nv) defm = solver.solve(B).eval();
			else defm(computeIdx, Eigen::all) = solver.solve(B(computeIdx, Eigen::all)).eval();

			// step 4
			U = U + defm - rest - Z;
		}

		return defm;
		#endif
	}


};

// test
#if false

Eigen::Matrix3d computeRotation(const Eigen::Matrix3d& cov) {
	#if true
	Eigen::JacobiSVD<Eigen::Matrix3d> svds(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

	auto& U = svds.matrixU();
	auto& V = svds.matrixV();
	Eigen::Vector3d D{ 1., 1., (V * U.transpose()).determinant() };
	return V * D.asDiagonal() * U.transpose();
	#else
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigs;
	eigs.computeDirect(cov.transpose() * cov, Eigen::ComputeEigenvectors);

	Eigen::Vector3d s = eigs.eigenvalues().cwiseSqrt().cwiseInverse();
	return (cov * eigs.eigenvectors() * s.asDiagonal()	* eigs.eigenvectors().transpose()).transpose();
	#endif
}

std::vector<Eigen::Matrix3d> arapRot(
	const kika::heds::SurfaceMesh& mesh,
	const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
	const Eigen::MatrixX3d& src,
	const Eigen::MatrixX3d& trg, const std::vector<Eigen::Matrix3d>& oldRot) {
	std::vector<Eigen::Matrix3d> R = oldRot;
	std::vector<Eigen::Matrix3d> C(src.rows(), Eigen::Matrix3d{});
	for (size_t i = 0; i < R.size(); i++) {
		Eigen::Matrix3d S = Eigen::Matrix3d::Zero();

		double coeff = 0.2 * vareas[i];
		Eigen::Matrix3d plus = Eigen::Matrix3d::Zero();
		size_t counter = 0;

		for (Eigen::SparseMatrix<double>::InnerIterator it(lbo, i); it; ++it) {
			if (it.row() == it.col()) continue;
			Eigen::Vector3d s = src.row(it.row()) - src.row(i);
			Eigen::Vector3d t = trg.row(it.row()) - trg.row(i);
			S += -it.value() * s * t.transpose();

			plus -= coeff * it.value() * R[it.row()].transpose(); // solid effect (original smooth enhance)
			// plus += coeff * it.value() * R[it.row()]; // shrink effect (small alpha 0.04)
			counter++;
		}

		// 
		// auto&& vs = mesh.verts[i]->oneringNeighbs();
		// for (size_t i = 0, k = vs.size(); i < k; i++) {
		// 	auto v = vs[i], u = vs[(i+1)%k];
		// 	double w = lbo.coeff(v, u);
		//
		// 	Eigen::Vector3d s = src.row(u) - src.row(v);
		// 	Eigen::Vector3d t = trg.row(u) - trg.row(v);
		// 	S += - w * s * t.transpose();
		// }

		C[i] = S + plus;
		R[i] = computeRotation(C[i]);
	}

	return R;
}

////	mesh.pos = kika::def::arap(mesh, defm, dsl.lbo, mesh.pos, krings, 
//// defm(test, Eigen::all), test, 20);
Eigen::MatrixX3d arap(const kika::heds::SurfaceMesh& mesh,
	const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
	const Eigen::MatrixX3d& defm,
	const Eigen::MatrixX3d& rest, const std::vector<uint32_t>& compIdx,
	const Eigen::MatrixX3d& ctrl, const std::vector<uint32_t>& ctrlIdx,
	const size_t maxIter=10) {

	std::vector<uint32_t> fixedIdx = kika::util::complement(mesh.getNumVerts(), compIdx);
	fixedIdx.insert(fixedIdx.end(), ctrlIdx.begin(), ctrlIdx.end());
	std::vector<uint32_t> movedIdx = kika::util::complement(mesh.getNumVerts(), fixedIdx);

	std::vector<bool> uncompute(mesh.getNumVerts(), false);
	for (auto i : fixedIdx) uncompute[i] = true;

	Eigen::SparseMatrix<double> A = lbo;
	A = kika::eig::spsub(lbo, movedIdx);
	// kika::eig::spfix(A, fixedIdx, 1.);

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);

	Eigen::MatrixX3d next = defm, init = rest;
	init(ctrlIdx, Eigen::all) = ctrl;
	std::vector<Eigen::Matrix3d> R(mesh.getNumVerts(), Eigen::Matrix3d::Identity());
	Eigen::MatrixX3d b(defm.rows(), 3);

	Eigen::MatrixX3d fixedLhs = Eigen::MatrixX3d::Zero(defm.rows(), 3);
	for (const auto& i : movedIdx) {
		auto&& onering = mesh.verts[i]->oneringNeighbs();
		for (size_t j = 0; j < onering.size(); j++) {
			if (uncompute[onering[j]]) {
				fixedLhs.row(i) -= lbo.coeff(i, onering[j]) * init.row(onering[j]);
			}
		}
	}

	for (size_t step = 0; step < maxIter; step++) {
		b.setZero();

		R = arapRot(mesh, lbo, vareas, rest, next, R);

		for (size_t i = 0; i < (size_t)defm.rows(); i++) {
			if (uncompute[i]) {
				b.row(i) = init.row(i);
			} else {

				// for (Eigen::SparseMatrix<double>::InnerIterator it(lbo, i); it; ++it) {
				// 	if (it.row() == it.col()) continue;
				//
				// 	double w = it.value();
				// 	b.row(i) += (1./6.) * w * (prev.row(it.row()) - prev.row(i)) * (2 * R[i] + 2 * R[it.row()] + R[mesh.halfedge(i, it.row())->next->vert->id] + R[mesh.halfedge(it.row(), i)->next->vert->id]).transpose();
				// }

				for (Eigen::SparseMatrix<double>::InnerIterator it(lbo, i); it; ++it) {
					if (it.row() == it.col()) continue;

					double w = it.value();
					b.row(i) += 0.5 * w * (rest.row(it.row()) - rest.row(i)) * (R[i] + R[it.row()]).transpose();
				}
			}
		}

		b += fixedLhs;
		Eigen::MatrixX3d tmp = solver.solve(b(movedIdx, Eigen::all));
		next(movedIdx, Eigen::all) = tmp;
		// next = solver.solve(b);
	}

	return next;
}
#endif

}

