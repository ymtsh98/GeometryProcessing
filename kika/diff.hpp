
//// ARAP shape deformer
//// Local Deformation for Interactive Shape Editing (2023)
template <bool SpokesAndRims = false>
class LocalARAPDeformer {
	std::vector<std::vector<uint32_t>> onerings; // counter-clock order onering neighbs
	std::vector<Eigen::VectorXd> wspokes, wrims; // weight of spokes, rims
	std::vector<Eigen::Matrix3d> rot; // vertex-wise rotation matrices
	Eigen::VectorXd vareas; // vertex areas

	double w, s; // region of influence
	double r, eps = 1e-6; // r = f(w, s, vareas) + eps;
	double smc; // smooth enhance coeff

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

	std::vector<uint32_t> computeIdx; // compute region index
	std::vector<uint32_t> controlIdx; // hard control index
	std::vector<uint32_t> boundaryIdx; // index of between compute and noncompute
	std::vector<uint32_t> full2comp; // full mesh index to compute region index
	std::vector<bool> uncompute; // is i-th vertex computed

	bool classicARAP;

public:

	LocalARAPDeformer() : w(1.), s(1.), r(1.), smc(0.04), classicARAP(false) {}

	// @param computeVertIdx : extra selection of compute region ((~computeVertIdx & controlVertIdx) will be fixed)
	// @param controlVertIdx : hard constraint control region
	template <typename T = int>
	LocalARAPDeformer(const kika::heds::SurfaceMesh& mesh, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
		const double w = 1.0, const double s = 0.001, const std::vector<T>& computeVertIdx ={}, const std::vector<T>& controlVertIdx ={})
		: w(1.), s(1.), r(1.), smc(0.04), classicARAP(false) {

		build(mesh, lbo, vareas, w, s, computeVertIdx, controlVertIdx);
	}

	// build deformer
	template <typename T = int>
	void build(const kika::heds::SurfaceMesh& mesh, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
		const double w = 1.0, const double s = 0.001, const std::vector<T>& computeVertIdx ={}, const std::vector<T>& controlVertIdx ={}) {
		updateWeights(mesh, lbo);
		updateLinearSolver(lbo, vareas, w, s, computeVertIdx, controlVertIdx);
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

private:

	// update index maps
	template <typename T = int>
	void updateIndexMap(const std::vector<T>& computeVertIdx ={}, const std::vector<T>& controlVertIdx ={}) {
		size_t nv = onerings.size();

		if (computeVertIdx.size() == 0) {
			computeIdx.resize(nv);
			std::iota(computeIdx.begin(), computeIdx.end(), 0);
		} else {
			computeIdx = kika::util::vecast<uint32_t>(computeVertIdx);
		}

		controlIdx = kika::util::vecast<uint32_t>(controlVertIdx);

		uncompute = std::vector<bool>(nv, true);
		for (auto i : computeIdx) uncompute[i] = false;
		for (auto i : controlIdx) uncompute[i] = true;

		computeIdx.clear();
		boundaryIdx.clear();
		full2comp = std::vector<uint32_t>(nv, std::numeric_limits<uint32_t>::max());
		std::vector<bool> added(nv, false);

		for (size_t i = 0; i < nv; i++) {
			if (!uncompute[i]) {
				full2comp[i] = (uint32_t)computeIdx.size();
				computeIdx.emplace_back((uint32_t)i);
				added[i] = true;
			}
		}

		// collect boundary(compute-uncompute bride)
		for (const auto i : computeIdx) {
			for (const auto j : onerings[i]) {
				if (!added[j] && uncompute[j]) {
					full2comp[j] = (uint32_t)(computeIdx.size() + boundaryIdx.size());
					boundaryIdx.emplace_back((uint32_t)j);
					added[j] = true;
				}
			}
		}

	}

public:

	// update global position solver
	template <typename T = int>
	void updateLinearSolver(const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& vareas,
		const double w = 1.0, const double s = 0.001, const std::vector<T>& computeVertIdx ={}, const std::vector<T>& controlVertIdx ={}) {
		size_t nv = onerings.size();

		updateIndexMap(computeVertIdx, controlVertIdx);

		this->w = w;
		this->s = s;
		this->vareas = vareas;

		r = (w * vareas.array()).matrix().maxCoeff() / s + eps;
		if (classicARAP) r = 0.;

		if (computeIdx.size() == nv) solver.compute(lbo + r * kika::eig::speye(vareas.rows()));
		else solver.compute(kika::eig::spsub(lbo + r * kika::eig::speye(vareas.rows()), computeIdx));
	}

	// update smooth enhance coeff
	void updateSmoothEnhanceCoeff(const double coeff) {
		this->smc = coeff;
	}

	// use classic ARAP methods
	// you can not use soft constraint in the same time
	void useClassicARAP(const bool use) {
		classicARAP = use;
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

	// debug all position version
	#if false

	// compute per-vertex rotation matrix
	void localRotations(const Eigen::MatrixX3d& defm, const Eigen::MatrixX3d& rest) {
		Eigen::VectorXd norm = (defm - rest).rowwise().squaredNorm();
		int nv = (int)norm.rows();

		#pragma omp parallel for
		for (int i = 0; i < nv; i++) {
			if (uncompute[i] || (norm(i) < eps && (norm(onerings[i]).array() < eps).count() == onerings[i].size())) {
				rot[i].setIdentity();
				continue;
			}

			Eigen::MatrixX3d dX = defm(onerings[i], Eigen::all), rX = rest(onerings[i], Eigen::all);

			// spokes term
			Eigen::MatrixX3d dE = dX.array().rowwise() - defm.row(i).array();
			Eigen::MatrixX3d rE = rX.array().rowwise() - rest.row(i).array();
			Eigen::Matrix3d cov = -rE.transpose() * wspokes[i].asDiagonal() * dE;

			// rims term
			if constexpr (SpokesAndRims) {
				auto circuit = kika::eig::circuitMat(onerings[i].size(), -1., 1.);
				dE = circuit * dX;
				rE = circuit * rX;
				cov += -rE.transpose() * wrims[i].asDiagonal() * dE;
			}

			// smooth rotation enhance
			double coeff = smc * vareas[i];
			if (smc < 0.) {
				for (size_t j = 0; j < onerings[i].size(); j++) {
					cov -= coeff * wspokes[i][j] * rot[onerings[i][j]];
				}
			} else if (smc > 0.) {
				for (size_t j = 0; j < onerings[i].size(); j++) {
					cov -= coeff * wspokes[i][j] * rot[onerings[i][j]].transpose();
				}
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
		int nv = (int)rest.rows();
		Eigen::MatrixX3d rhs = Eigen::MatrixX3d::Zero(nv, 3);

		#pragma omp parallel for
		for (int i = 0; i < nv; i++) {
			if (uncompute[i]) continue;

			for (size_t j = 0, k = onerings[i].size(); j < k; j++) {
				if constexpr (SpokesAndRims) {
					double denom = 4.;
					Eigen::Matrix3d R = 2. * rot[onerings[i][j]] + 2. * rot[i];

					// boudnary condition (this is lazy condition), just skip
					auto v = onerings[i][(j + onerings[i].size() - 1) % k], u = onerings[i][(j + 1) % k];
					if (wrims[i][(j + onerings[i].size() - 1) % k] != 0.) { R += rot[v]; denom += 1.; }
					if (wrims[i][(j + 1) % k] != 0.) { R += rot[u]; denom += 1.; }

					rhs.row(i) += (1. / denom) * wspokes[i][j] * R * (rest.row(onerings[i][j]) - rest.row(i)).transpose();
				} else {
					rhs.row(i) += 0.5 * wspokes[i][j] * (rot[onerings[i][j]] + rot[i]) * (rest.row(onerings[i][j]) - rest.row(i)).transpose();
				}
			}
		}

		return rhs;
	}

public:

	// WARNINGS : position of softControlIdx will not be preserved
	// @param ctrl : controller position matrix |controlIdx| x 3 (if you set updateLinearSolver controlVertIdx, you can set hard constraint)
	// @param softControlIdx : controller vertex index (this is soft constraint)
	template <typename T = std::vector<uint32_t>>
	Eigen::MatrixX3d run(Eigen::MatrixX3d defm, const Eigen::MatrixX3d& rest, const size_t iter, const T& softControlIdx ={}) {

		size_t nv = (size_t)defm.rows();

		rot.resize(nv);
		std::fill(rot.begin(), rot.end(), Eigen::Matrix3d::Identity());

		Eigen::MatrixX3d U = Eigen::MatrixX3d::Zero(nv, 3);

		Eigen::MatrixX3d fixedLhs = computeFixedLhs(defm);
		Eigen::MatrixX3d ctrl = defm(softControlIdx, Eigen::all);

		for (size_t step = 0; step < iter; step++) {
			// soft constrained controller position
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
	}

	#else

private:

	// compute per-vertex rotation matrix
	void localRotations(const Eigen::MatrixX3d& defm, const Eigen::MatrixX3d& rest) {
		Eigen::VectorXd norm = (defm - rest).rowwise().squaredNorm();

		int n = (int)computeIdx.size();
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			int ci = computeIdx[i];

			std::vector<uint32_t> onering(onerings[ci].size());
			for (size_t j = 0; j < onering.size(); j++) onering[j] = full2comp[onerings[ci][j]];

			if (uncompute[ci] || (norm(i) < eps && (norm(onering).array() < eps).count() == onering.size())) {
				rot[i].setIdentity();
				continue;
			}

			Eigen::VectorXd spokes = wspokes[ci];

			Eigen::MatrixX3d dX = defm(onering, Eigen::all), rX = rest(onering, Eigen::all);

			// // spokes term
			Eigen::MatrixX3d dE = dX.array().rowwise() - defm.row(i).array();
			Eigen::MatrixX3d rE = rX.array().rowwise() - rest.row(i).array();
			Eigen::Matrix3d cov = -rE.transpose() * spokes.asDiagonal() * dE;

			// rims term
			if constexpr (SpokesAndRims) {
				Eigen::VectorXd rims = wrims[ci];
				auto circuit = kika::eig::circuitMat(onering.size(), -1., 1.);
				dE = circuit * dX;
				rE = circuit * rX;
				cov += -rE.transpose() * rims.asDiagonal() * dE;
			}

			// smooth rotation enhance
			double coeff = smc * vareas[ci];
			if (smc < 0.) {
				for (size_t j = 0; j < onering.size(); j++) {
					cov -= coeff * spokes[j] * rot[onering[j]];
				}
			} else if (smc > 0.) {
				for (size_t j = 0; j < onering.size(); j++) {
					cov -= coeff * spokes[j] * rot[onering[j]].transpose();
				}
			}

			rot[i] = computeRotation(cov);
		}
	}

	// SC-L1
	Eigen::MatrixX3d smoothlyClampedL1Norm(const Eigen::MatrixX3d& defm, const Eigen::MatrixX3d& rest, const Eigen::VectorXd& vareasRestrict, const Eigen::MatrixX3d& U) {
		Eigen::MatrixX3d Z = defm - rest + U;

		Eigen::VectorXd znorm = Z.rowwise().norm().cwiseMax(1e-20);
		Eigen::VectorXd shrink = ((r * s - w * vareasRestrict.array() * (s / znorm.array())) / (r * s - w * vareasRestrict.array())).cwiseMax(0);
		Eigen::VectorXd local = (znorm.array() > s).select(1., shrink.array());
		Z = local.asDiagonal() * Z;
		return Z;
	}

	// compute original ARAP rhs
	Eigen::MatrixX3d arapRhs(const Eigen::MatrixX3d& rest) {
		int n = (int)computeIdx.size();
		Eigen::MatrixX3d rhs = Eigen::MatrixX3d::Zero(n + boundaryIdx.size(), 3);

		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			auto ci = computeIdx[i];
			if (uncompute[ci]) continue;

			std::vector<uint32_t> onering(onerings[ci].size());
			Eigen::VectorXd spokes = wspokes[ci];
			for (size_t j = 0; j < onering.size(); j++) {
				onering[j] = full2comp[onerings[ci][j]];
			}

			for (size_t j = 0, k = onering.size(); j < k; j++) {
				if constexpr (SpokesAndRims) {
					Eigen::VectorXd rims = wrims[ci];

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
	// @param defm : if you set controlVertIdx in updateLinearSolver, used defm hard constraint
	// @param softControlIdx : controller vertex index (this is soft constraint)
	template <typename T = std::vector<uint32_t>>
	Eigen::MatrixX3d run(Eigen::MatrixX3d defm, const Eigen::MatrixX3d& rest, const size_t iter, const T& softControlIdx ={}) {

		std::vector<uint32_t> restrictIdx(computeIdx.begin(), computeIdx.end());
		restrictIdx.insert(restrictIdx.end(), boundaryIdx.begin(), boundaryIdx.end());

		size_t rv = restrictIdx.size();

		rot.resize(rv);
		std::fill(rot.begin(), rot.end(), Eigen::Matrix3d::Identity());

		Eigen::MatrixX3d U = Eigen::MatrixX3d::Zero(rv, 3);

		Eigen::MatrixX3d fixedLhs = computeFixedLhs(defm);
		Eigen::MatrixX3d restRestrict = rest(restrictIdx, Eigen::all);
		Eigen::VectorXd vareasRestrict = vareas(restrictIdx, Eigen::all);
		Eigen::MatrixX3d defmRestrict = defm(restrictIdx, Eigen::all);
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
			// soft constrained controller position
			if (softControlIdxRestrict.size()) defmRestrict(softControlIdxRestrict, Eigen::all) = ctrlRestrict;

			// step 1
			if (step != 0) localRotations(defmRestrict, restRestrict);

			// // step 2
			Eigen::MatrixX3d Z = smoothlyClampedL1Norm(defmRestrict, restRestrict, vareasRestrict, U);
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
	}
	#endif

};