#pragma once
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>

#include "kernel.hpp"
#include "util.hpp"

namespace cpd {

	constexpr double pi = 3.141592653589;

	// gram matrix type
	enum class GramType {
		FULL = 0,
		NYSTROM = 1,
	};

	// coherent point drift base class
	template <typename Derived1, typename Derived2,	typename Prec = typename Derived1::Scalar>
	class Base {

	protected:
		using CPDMatrix = Eigen::Matrix<Prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
		using CPDVector = Eigen::Matrix<Prec, Eigen::Dynamic, 1, Eigen::ColMajor>;

		int n, m, dim;
		std::reference_wrapper<const Eigen::MatrixBase<Derived1>> Y;
		std::reference_wrapper<const Eigen::MatrixBase<Derived2>> X;
		CPDMatrix Yt;

		CPDMatrix PX;
		CPDVector P1, PT1;
		Prec pn;

		Prec weight;
		Prec variance;
		size_t iter;

		Base(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg, const Prec weight, const Prec gamma) :
			n((int)trg.rows()), m((int)src.rows()), dim((int)src.cols()),
			Y(src), X(trg), Yt(src),
			PX(m, dim), P1(m), PT1(n), pn(0.0),
			weight(weight), iter(0) {

			size_t k1 = static_cast<size_t>(sqrt(n)), k2 = static_cast<size_t>(sqrt(m));
			variance = gamma * pairwiseSquaredNorm(randomSampling(X.get(), k1), randomSampling(Yt, k2)) / (dim * k1 * k2);
		}

		virtual ~Base() {}

		virtual void estimationStep() {
			Prec sigma = 2. * variance;
			Prec omega = pow(cpd::pi * sigma, dim/2.0) * (weight / (1.0 - weight)) * ((Prec)m / n);

			PX.fill(0.0);
			P1.fill(0.0);

			#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				CPDVector phi = Eigen::exp(-(Yt.rowwise() - X.get().row(i)).rowwise().squaredNorm().array() / sigma);
				phi.array() /= phi.sum() + omega;

				PT1(i) = phi.sum();
				CPDMatrix PXt = phi * X.get().row(i);
				#pragma omp critical
				{
					P1 += phi;
					PX += PXt;
				}
			}

			pn = P1.sum();
		}

		//TODO
		virtual void estimationStepNystrom(const size_t kest = 200) {
			Prec omega = pow(cpd::pi * 2 * variance, dim/2.0) * (weight / (1.0 - weight)) * ((Prec)m / n);

			Prec sqvar = sqrt(variance);
			
			CPDMatrix U = randomSampling(vConcatDenseMat(Yt, X.get()), kest);

			CPDMatrix Kyu = gaussianKernel(Yt, U, sqvar);
			CPDMatrix Kuu = gaussianKernel(U, sqvar);
			CPDMatrix Kux = gaussianKernel(U, X.get(), sqvar);
			
			Eigen::JacobiSVD<CPDMatrix> SVD(Kuu, Eigen::ComputeThinV);
			auto V = SVD.matrixV();
			
			CPDVector Guu = SVD.singularValues();
			CPDMatrix Gyu = Kyu * (sqrt((Prec)kest / m) * V * Guu.asDiagonal().inverse());
			CPDMatrix Gux = (sqrt((Prec)kest / n) * V * Guu.asDiagonal().inverse()).transpose() * Kux;
			
			Guu = (Prec)m * n / (kest * kest) * Guu;
			
			CPDVector alpha = 1.0 / ((Gux.transpose() * (Guu.asDiagonal() * (Gyu.transpose() * CPDVector::Ones(n)))).array() + omega);
			PT1 = 1.0 - omega * alpha.array();
			P1 = Gyu * (Guu.asDiagonal() * (Gux * alpha));
			PX = Gyu * (Guu.asDiagonal() * (Gux * static_cast<CPDMatrix>(X.get().array().colwise() * alpha.array())));
		}

		//TODO
		virtual void estimationStepKdtree() {}

		//TODO
		virtual void estimationStepIfgt() {}

		virtual void maximizationStep() {}

		virtual void displayParams() {}

	public:
		// run single iteration
		void update(const bool displayParameters = false) {
			//TODO
			// if (variance > nystromSigma) estimationStepNystrom(kest);
			// else if (variance < kdtreeSigma) estimationStepKdtree(radius);
			// else estimationStep();

			estimationStep();
			maximizationStep();
			iter++;
			if (displayParameters) displayParams();
		}

		CPDMatrix getTransformedPoints() {
			return Yt;
		}

		// get the prediction corresponding to ith point
		CPDVector getPrediction(const size_t i) {
			Prec sigma = 2.0 * variance;

			return Eigen::exp(-(X.get().rowwise() - Yt.row(i)).rowwise().squaredNorm().array() / sigma) /
				(Eigen::exp(-(X.get().rowwise() - Yt.row(i)).rowwise().squaredNorm().array() / sigma).sum() +
				 pow(cpd::pi * sigma, dim/2.0) * (weight / (1.0 - weight)) * ((Prec)m / n));
		}
	};

	// rigid transform cpd
	template <typename Derived1, typename Derived2, typename Prec = typename Derived1::Scalar>
	class Rigid final  : public Base<Derived1, Derived2, Prec> {
		using BaseType = Base<Derived1, Derived2, Prec>;
		using CPDMatrix = typename BaseType::CPDMatrix;
		using CPDVector = typename BaseType::CPDVector;
		using BaseType::n, BaseType::m, BaseType::dim,
			BaseType::Y, BaseType::X, BaseType::Yt,
			BaseType::PX, BaseType::P1, BaseType::PT1, BaseType::pn,
			BaseType::weight, BaseType::variance, BaseType::iter;

		CPDMatrix R;
		CPDVector t;
		Prec s;

		void maximizationStep() {
			CPDVector Xmean = X.get().transpose() * PT1 / pn;
			CPDVector Ymean = Y.get().transpose() * P1 / pn;

			CPDMatrix Xhat = X.get().rowwise() - Xmean.transpose();
			CPDMatrix	Yhat = Y.get().rowwise() - Ymean.transpose();

			CPDMatrix A = (PX - P1 * Xmean.transpose()).transpose() * Yhat;

			Eigen::JacobiSVD<CPDMatrix> SVD(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto V = SVD.matrixV();
			auto U = SVD.matrixU();

			CPDVector C = CPDVector::Ones(dim);
			C(dim-1) = (U * V.transpose()).determinant();
			R = U * C.asDiagonal() * V.transpose();

			Prec trATR = (A.transpose() * R).trace();
			s = trATR / (Yhat.transpose() * P1.asDiagonal() * Yhat).trace();

			t = Xmean - s * R * Ymean;

			variance = ((Xhat.transpose() * PT1.asDiagonal() * Xhat).trace() - s * trATR) / (pn * dim);

			Yt = transform(Y.get());
		}

		void displayParams() {
			std::stringstream ss;
			ss << "rigid cpd iterations : " << iter << "\n";
			ss << "rotation" << "\n";
			ss << R << "\n";
			ss << "translation" << "\n";
			ss << t.transpose() << "\n";
			ss << "scaling" << "\n";
			ss << s << "\n" << "\n";
			std::cout << ss.str();
		}

	public:

		// @param src	: source point set MxD
		// @param trg	: target point set NxD
		// @param weight	: amount of noise in the point sets (0.0 <= weight < 1.0)
		// @param gamma	: randomness of the initial variance (0.0 < gamma)
		Rigid(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg, const Prec weight = 0.2, const Prec gamma = 1.0) :
			Base<Derived1, Derived2, Prec>(src, trg, weight, gamma), R(CPDMatrix::Identity(dim, dim)), t(CPDVector::Zero(dim)), s(1.0) {}

		// apply transformation input point set
		// @param V : source point set
		// @return transformed point set
		template <typename Derived>
		CPDMatrix transform(const Eigen::MatrixBase<Derived>& V) {
			return (s * V * R.transpose()).rowwise() + t.transpose();
		}

		///	Transform(Y) = (s * Y * R.transpose()).rowwise() + t.transpose();
		///	@return R : rotation matrix DxD,  
		///	        t : translation vector Dx1, 
		///	        s : scaling value 1x1
		std::tuple<CPDMatrix, CPDVector, Prec> getTransformations() {
			return std::make_tuple(R, t, s);
		}
	};

	// affine transform cpd
	template <typename Derived1, typename Derived2, typename Prec = typename Derived1::Scalar>
	class Affine final : public Base<Derived1, Derived2, Prec> {
		using BaseType = Base<Derived1, Derived2, Prec>;
		using CPDMatrix = typename BaseType::CPDMatrix;
		using CPDVector = typename BaseType::CPDVector;
		using BaseType::n, BaseType::m, BaseType::dim,
			BaseType::Y, BaseType::X, BaseType::Yt,
			BaseType::PX, BaseType::P1, BaseType::PT1, BaseType::pn,
			BaseType::weight, BaseType::variance, BaseType::iter;

		CPDMatrix B;
		CPDVector t;

		void maximizationStep() {
			CPDVector Xmean = X.get().transpose() * PT1 / pn;
			CPDVector Ymean = Y.get().transpose() * P1 / pn;

			CPDMatrix Xhat = X.get().rowwise() - Xmean.transpose();
			CPDMatrix	Yhat = Y.get().rowwise() - Ymean.transpose();

			CPDMatrix A = (PX - P1 * Xmean.transpose()).transpose() * Yhat;

			B = A * (Yhat.transpose() * P1.asDiagonal() * Yhat).inverse();
			t = Xmean - B * Ymean;

			variance = ((Xhat.transpose() * PT1.asDiagonal() * Xhat).trace() -
									(A * B.transpose()).trace()) / (pn * dim);

			Yt = transform(Y.get());
		}

		void displayParams() {
			std::stringstream ss;
			ss << "affine cpd iterations : " << iter << "\n";
			ss << "affine transformation" << "\n";
			ss << B << "\n";
			ss << "translation" << "\n";
			ss << t.transpose() << "\n" << "\n";
			std::cout << ss.str();
		}

	public:

		// @param src	: source point set MxD
		// @param trg	: target point set NxD
		// @param weight	: amount of noise in the point sets (0.0 <= weight < 1.0)
		// @param gamma	: randomness of the initial variance (0.0 < gamma)
		Affine(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg, const Prec weight = 0.2, const Prec gamma = 1.0) :
			Base<Derived1, Derived2, Prec>(src, trg, weight, gamma), B(CPDMatrix::Identity(dim, dim)), t(CPDVector::Zero(dim)) {}

		// apply transformation input point set
		// @param V : source point set
		// @return transformed point set
		template <typename Derived>
		CPDMatrix transform(const Eigen::MatrixBase<Derived>& V) {
			return (V * B.transpose()).rowwise() + t.transpose();
		}

		// Transform(Y) = (Y * B.transpose()).rowwise() + t.transpose();
		// @return B : affine matrix DxD, 
		//         t : translation vector Dx1
		std::tuple<CPDMatrix, CPDVector> getTransformations() {
			return std::make_tuple(B, t);
		}
	};

	// nonrigid transform cpd
	template <cpd::GramType GT = cpd::GramType::NYSTROM, typename Derived1 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
		typename Derived2 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, typename Prec = typename Derived1::Scalar>
	class Nonrigid final : public Base<Derived1, Derived2, Prec> {
		using BaseType = Base<Derived1, Derived2, Prec>;
		using CPDMatrix = typename BaseType::CPDMatrix;
		using CPDVector = typename BaseType::CPDVector;
		using BaseType::n, BaseType::m, BaseType::dim,
			BaseType::Y, BaseType::X, BaseType::Yt,
			BaseType::PX, BaseType::P1, BaseType::PT1, BaseType::pn,
			BaseType::weight, BaseType::variance, BaseType::iter;

		CPDMatrix W;
		Prec lambda;

		CPDMatrix G;
		CPDMatrix Q, QL, QT;
		CPDVector Linv;

		void maximizationStep() {
			CPDVector P1inv = 1.0 / P1.array();
			if (!P1inv.allFinite()) replaceInf2ValidNum(P1inv, std::numeric_limits<Prec>::max());

			if constexpr (GT == cpd::GramType::FULL) {
				Eigen::LLT<CPDMatrix> solver(G + (lambda * variance * P1inv.asDiagonal()).toDenseMatrix());
				W = G * solver.solve(P1inv.asDiagonal() * PX - Y.get());

			} else if constexpr (GT == cpd::GramType::NYSTROM) {
				Prec c = 1.0 / lambda * variance;
				CPDMatrix rhs = P1inv.asDiagonal() * PX - Y.get();
				CPDMatrix tmp = c * QT * P1.asDiagonal() * Q;
				tmp += Linv.asDiagonal();

				W = QL * (QT * c * P1.asDiagonal() * (rhs - c * Q *
									(tmp.inverse() * (QT * (P1.asDiagonal() * rhs)))));

			}

			Yt = transform(Y.get());

			variance = ((X.get().transpose() * PT1.asDiagonal() * X.get()).trace() -
									2 * (PX.transpose() * Yt).trace() +
									(Yt.transpose() * P1.asDiagonal() * Yt).trace()) / (pn * dim);
		}

		void displayParams() {
			std::stringstream ss;
			ss << "nonrigid cpd iterations : " << iter << "\n";
			ss << "nonrigid transform (i=0, 1, 2)" << "\n";
			ss << W.block(0, 0, std::min(3, m), dim) << "\n" << "\n";
			std::cout << ss.str();
		}

	public:
		// @param src	: source point set MxD
		// @param trg	: target point set NxD
		// @param weight	: amount of noise in the point sets (0.0 <= weight < 1.0)
		// @param gamma	: randomness of the initial variance (0.0 < gamma)
		// @param kgram	: sample size of gram matrix approximation (if kgram == 0, then kgram = M^(2/5))
		// @param lambda	: goodness of maximum likelihood fit and regularization (0.0 < lambda)
		// @param beta	: width of smoothing gaussian filter (0.0 < beta)
		Nonrigid(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg, const Prec weight = 0.2, const Prec gamma = 1.0,
							size_t kgram = 0, const Prec lambda = 2.0, const Prec beta = 2.0) :
			Base<Derived1, Derived2, Prec>(src, trg, weight, gamma),
			W(CPDMatrix::Zero(m, dim)), lambda(lambda) {

			if constexpr (GT == cpd::GramType::FULL) {
				G = gaussianKernel(Y.get(), beta);

			} else if constexpr (GT == cpd::GramType::NYSTROM) {
				if (kgram == 0) kgram = static_cast<size_t>(pow(m, 2./5.));

				auto randomIndices = cpd::randomPermutation(m);

				std::tie(Q, Linv) = cpd::nystromApprox(sortByIndices(
					gaussianKernel(Y.get(), beta, std::vector<size_t>(randomIndices.begin(), randomIndices.begin() + kgram)),
					randomIndices, false, 1));

				QT = sortByIndices(Q.transpose(), randomIndices, true, 0);
				Q = sortByIndices(Q, randomIndices, true, 1);
				QL = Q * Linv.asDiagonal();
				Linv = 1.0 / Linv.array();
			}
		}

		// apply transformation input point set
		// @param V : source point set
		// @return transformed point set
		template <typename Derived>
		CPDMatrix transform(const Eigen::MatrixBase<Derived>& V) {
			return V + W;
		}

		// Transform(Y) = Y + W;
		// @return W : displacement matrix MxD
		CPDMatrix getTransformations() {
			return W;
		}
	};

	// bayesian cpd
	template <cpd::GramType GT = cpd::GramType::NYSTROM, typename Derived1 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
		typename Derived2 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, typename Prec = typename Derived1::Scalar>
	class Bayesian : public Base<Derived1, Derived2, Prec> {
	protected:
		using BaseType = Base<Derived1, Derived2, Prec>;
		using CPDMatrix = typename BaseType::CPDMatrix;
		using CPDVector = typename BaseType::CPDVector;
		using BaseType::n, BaseType::m, BaseType::dim,
			BaseType::Y, BaseType::X, BaseType::Yt,
			BaseType::PX, BaseType::P1, BaseType::PT1, BaseType::pn,
			BaseType::weight, BaseType::variance, BaseType::iter;

		CPDMatrix W;
		CPDMatrix R;
		CPDVector t;
		Prec s;

		Prec lambda, kappa;

		CPDMatrix Ginv;
		CPDMatrix Q, QL, QT;
		CPDVector Linv;

		using SigmaMatrix = std::conditional_t<GT==cpd::GramType::NYSTROM,
			Eigen::DiagonalMatrix<Prec, Eigen::Dynamic, Eigen::Dynamic>, CPDMatrix>;

		SigmaMatrix Sigma;
		CPDVector Alpha;

		void estimationStep() {
			Prec sigma = 2.0 * variance;
			CPDVector c = pow(cpd::pi * sigma, -dim / 2.0) *
				Eigen::exp(-s * s * Sigma.diagonal().array() * dim / sigma);
			CPDVector alpha = (1.0 - weight) * Alpha.array();

			P1.fill(0.0);
			PX.fill(0.0);

			#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				CPDVector phi = c.array() * Eigen::exp(-(Yt.rowwise() - X.get().row(i)).rowwise().squaredNorm().array() / sigma);
				phi = alpha.array() * phi.array() / ((weight / n) + alpha.transpose() * phi);

				CPDMatrix PXt = phi * X.get().row(i);
				PT1(i) = phi.sum();
				#pragma omp critical
				{
					P1 += phi;
					PX += PXt;
				}
			}

			pn = P1.sum();
		}

		void estimationStepNystrom(const size_t kest = 200) {
			Prec sqvar = sqrt(variance);

			CPDMatrix U = randomSampling(vConcatDenseMat(Yt, X.get()), kest);

			CPDMatrix Kyu = gaussianKernel(Yt, U, sqvar);
			CPDMatrix Kuu = gaussianKernel(U, sqvar);
			CPDMatrix Kux = gaussianKernel(U, X.get(), sqvar);

			Eigen::JacobiSVD<CPDMatrix> SVD(Kuu, Eigen::ComputeThinV);
			auto V = SVD.matrixV();

			CPDVector Guu = SVD.singularValues();
			CPDMatrix Gyu = Kyu * (sqrt((Prec)kest / m) * V * Guu.asDiagonal().inverse());
			CPDMatrix Gux = (sqrt((Prec)kest / n) * V * Guu.asDiagonal().inverse()).transpose() * Kux;

			Guu = (Prec)m * n / (kest * kest) * Guu;

			Prec c = (weight / (1.0 - weight) / n) * pow(2.0 * cpd::pi * variance, dim / 2.0);
			CPDVector b = Alpha.array() *  Eigen::exp(- s * s / (2.0 * variance) * dim * Sigma.diagonal().array());
			CPDVector q = 1.0 / ((Gux.transpose() * (Guu.asDiagonal() * (Gyu.transpose() * b))).array() + c);
			PT1 = 1.0 - c * q.array();
			P1 = (Gyu * (Guu.asDiagonal() * (Gux * q))).array() * b.array();
			PX = (Gyu * (Guu.asDiagonal() * (Gux * static_cast<CPDMatrix>(X.get().array().colwise() * q.array())))).array().colwise() * b.array();
		}

		//TODO
		void estimationStepKdtree() {}

		//TODO
		void estimationStepIfgt() {}

		void maximizationStep() {
			CPDVector P1inv = 1.0 / P1.array();
			if (!P1inv.allFinite()) replaceInf2ValidNum(P1inv, std::numeric_limits<Prec>::max());

			CPDMatrix Xhat = P1inv.asDiagonal() * PX;

			Prec c = (s * s / variance);

			if constexpr (GT == cpd::GramType::FULL) {
				//Sigma = (Ginv + c * P1.asDiagonal().toDenseMatrix()).llt().solve(Eigen::MatrixXd::Identity(m, m));
				Sigma = (Ginv + c * P1.asDiagonal().toDenseMatrix()).inverse();
				W = (c *	Sigma * (P1.asDiagonal() * (invSimilarityTransform(Xhat) - Y.get())));

			} else if constexpr (GT == cpd::GramType::NYSTROM) {
				CPDMatrix S = QT* P1.asDiagonal()* Q;
				CPDMatrix tmp = S;
				tmp += (lambda / c) * Linv.asDiagonal();

				CPDMatrix Sr = (CPDMatrix::Identity(Linv.rows(), Linv.rows()) - S * tmp.inverse()) * QT;
				Sigma.diagonal() = (QL * Sr).diagonal() / lambda;
			
				W = ((c / lambda) *	QL * (Sr * (P1.asDiagonal() * (invSimilarityTransform(Xhat) - Y.get()))));
			}

			CPDMatrix Uhat = Y.get() + W;

			Alpha = Eigen::exp(Eigen::digamma(kappa + P1.array()) -
													Eigen::digamma(CPDVector::Ones(1).array() * (kappa * m + pn)).value());

			CPDVector Xmean = (Xhat.transpose() * P1) / pn;
			CPDVector Umean = (Uhat.transpose() * P1) / pn;
			Prec Smean = (Sigma.diagonal().transpose() * P1).value() / pn;

			CPDMatrix Sxu = (Xhat.rowwise() - Xmean.transpose()).transpose() * P1.asDiagonal() * (Uhat.rowwise() - Umean.transpose()) / pn;
			CPDMatrix Suu = (Uhat.rowwise() - Umean.transpose()).transpose() * P1.asDiagonal() * (Uhat.rowwise() - Umean.transpose()) / pn;
			Suu += Smean * CPDMatrix::Identity(dim, dim);

			Eigen::JacobiSVD<CPDMatrix> SVD(Sxu, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto V = SVD.matrixV();
			auto U = SVD.matrixU();

			CPDVector C = CPDVector::Ones(dim);
			C(dim-1) = (U * V.transpose()).determinant();
			R = U * C.asDiagonal() * V.transpose();

			s = (Sxu.transpose() * R).trace() / Suu.trace();
			t = Xmean - s * R * Umean;

			Yt = similarityTransform(Uhat);

			variance = ((X.get().transpose() * PT1.asDiagonal() * X.get()).trace() -
									2.0 * (PX.transpose() * Yt).trace() +
									(Yt.transpose() * P1.asDiagonal() * Yt).trace()) / (pn * dim) + s * s * Smean;
		}

		void displayParams() {
			std::stringstream ss;
			ss << "bayesian cpd iterations : " << iter << "\n";
			ss << "nonrigid transform (i=0, 1, 2)" << "\n";
			ss << W.block(0, 0, std::min(3, m), dim) << "\n";
			ss << "rotation" << "\n";
			ss << R << "\n";
			ss << "translation" << "\n";
			ss << t.transpose() << "\n";
			ss << "scaling" << "\n";
			ss << s << "\n" << "\n";
			std::cout << ss.str();
		}

		template <typename Derived>
		CPDMatrix similarityTransform(const Eigen::MatrixBase<Derived>& V) {
			return (s * V * R.transpose()).rowwise() + t.transpose();
		}

		template <typename Derived>
		CPDMatrix invSimilarityTransform(const Eigen::MatrixBase<Derived>& V) {
			return (1.0 / s) * ((V.rowwise() - t.transpose()) * R);
		}

	public:

		// @param src	: source point set MxD
		// @param trg	: target point set NxD
		// @param weight	: amount of noise in the point sets (0.0 <= weight < 1.0)
		// @param gamma	: randomness of the initial variance (0.0 < gamma)
		// @param kgram	: sample size of gram matrix approximation (if kgram == 0, then kgram = sqrt(M))
		// @param lambda	: goodness of maximum likelihood fit and regularization (0.0 < lambda)
		// @param beta	: width of smoothing gaussian filter (0.0 < beta)
		// @param kappa	: randomness of mixing coefficients (0.0 < kappa)
		Bayesian(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg, 
						 const Prec weight = 0.2, const Prec gamma = 1.0,
							size_t kgram = 0, const Prec lambda = 2.0, const Prec beta = 2.0, const Prec kappa = 1.0) :
			Base<Derived1, Derived2, Prec>(src, trg, weight, gamma),
			W(CPDMatrix::Zero(m, dim)),
			R(CPDMatrix::Identity(dim, dim)), t(CPDVector::Zero(dim)), s(1.0),
			lambda(lambda), kappa(kappa), Alpha(CPDVector::Ones(m)) {

			if constexpr (GT == cpd::GramType::FULL) {
				Ginv = gaussianKernel(Y.get(), beta);
				//Ginv = lambda * Ginv.llt().solve(Eigen::MatrixXd::Identity(m, m));
				Ginv = lambda * Ginv.inverse();
				Sigma = SigmaMatrix::Identity(m, m);

			} else if constexpr (GT == cpd::GramType::NYSTROM) {
				if (kgram == 0) kgram = static_cast<size_t>(sqrt(m));

				auto randomIndices = cpd::randomPermutation(m);

				std::tie(Q, Linv) = nystromApprox(sortByIndices(
					gaussianKernel(Y.get(), beta, std::vector<int>(randomIndices.begin(), randomIndices.begin() + kgram)),
					randomIndices, false, 1));

				QT = sortByIndices(Q.transpose(), randomIndices, true, 0);
				Q = sortByIndices(Q, randomIndices, true, 1);
				QL = Q * Linv.asDiagonal();
				Linv = 1.0 / Linv.array();

				Sigma = SigmaMatrix(m);
				Sigma.setIdentity();
			}
			Alpha *= 1.0 / m;
		}

		// HKS
		Bayesian(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg,
						 const Eigen::MatrixXd& evecs, const Eigen::MatrixXd& evals, 
						 const Prec weight = 0.2, const Prec gamma = 1.0,
						 size_t kgram = 0, const Prec lambda = 2.0, const Prec beta = 2.0, const Prec kappa = 1.0) :
			Base<Derived1, Derived2, Prec>(src, trg, weight, gamma),
			W(CPDMatrix::Zero(m, dim)),
			R(CPDMatrix::Identity(dim, dim)), t(CPDVector::Zero(dim)), s(1.0),
			lambda(lambda), kappa(kappa), Alpha(CPDVector::Ones(m)) {

			double min_param = abs(4.0 * log(10.0) / evals.maxCoeff());
			double max_param = abs(4.0 * log(10.0) / evals.block(1, 0, 1, 1).value());
			double step = log(max_param) - log(min_param);
			double time = exp(log(min_param) + step*0.25);
			Q = evecs.block(0, 1, evecs.rows(), evecs.cols()-1);
			QT = Q.transpose();
			Linv = Eigen::exp(-evals.block(1, 0, evals.rows()-1, 1).array() * time);
			QL = Q * Linv.asDiagonal();
			Linv = 1.0 / Linv.array();

			Sigma = SigmaMatrix(m);
			Sigma.setIdentity();
			
			Alpha *= 1.0 / m;
		}

		// apply transformation input point set
		// @param V : source point set
		// @return transformed point set
		template <typename Derived>
		CPDMatrix transform(const Eigen::MatrixBase<Derived>& V) {
			return similarityTransform(V + W);
		}

		// Transform(Y) = (s * (Y + W) * R.transpose()).rowwise() + t.transpose();
		// @return W : displacement matrix MxD, 
		//         R : rotation matrix DxD, 
		//         t : translation vector Dx1, 
		//         s : scaling value 1x1
		std::tuple<CPDMatrix, CPDMatrix, CPDVector, Prec> getTransformations() {
			return std::make_tuple(W, R, t, s);
		}

		// get the prediction corresponding to ith point
		CPDVector getPrediction(const size_t i) {
			Prec sigma = 2.0 * variance;
			double c = pow(cpd::pi * sigma, -dim / 2.0) *
				exp(-s * s * Sigma.diagonal()(i) * dim / sigma);

			double alpha = (1.0 - weight) * Alpha(i);

			CPDVector phi = c * Eigen::exp(-(X.get().rowwise() - Yt.row(i)).rowwise().squaredNorm().array() / sigma);
			return phi = alpha * phi.array() / ((weight / n) + (alpha * phi).sum());
		}
	};

	// down sampler
	template <typename Derived1, typename Derived2, typename Prec = typename Derived1::Scalar>
	class DownSampler {
	protected:
		size_t ky, kx;
		Eigen::Matrix<Prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Ysample, Xsample;

		DownSampler(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg, const size_t ky, const size_t kx) :
			ky(ky), kx(kx), Ysample(randomSampling(src, ky)), Xsample(randomSampling(trg, kx)) {}
	};

	// bayesian cpd with down sampliing acceleration
	template <cpd::GramType GT = cpd::GramType::NYSTROM, typename Derived1 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
		typename Derived2 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, typename Prec = typename Derived1::Scalar>
	class FastBayesian final : public DownSampler<Derived1, Derived2, Prec>, Bayesian<GT, Derived1, Derived2, Prec> {
	using BaseType = Bayesian<GT, Derived1, Derived2, Prec>;
	using CPDMatrix = typename BaseType::CPDMatrix;
	using CPDVector = typename BaseType::CPDVector;
	using BaseType::n, BaseType::m, BaseType::dim,
		BaseType::Y, BaseType::X, BaseType::Yt,
		BaseType::PX, BaseType::P1, BaseType::PT1, BaseType::pn,
		BaseType::weight, BaseType::variance, BaseType::iter;
	using BaseType::lambda, BaseType::s, BaseType::Q, BaseType::QL, BaseType::QT,
		BaseType::Linv, BaseType::similarityTransform, BaseType::invSimilarityTransform;

	std::reference_wrapper<const Eigen::MatrixBase<Derived1>> Yorig;
	std::reference_wrapper<const Eigen::MatrixBase<Derived2>> Xorig;
	Prec beta;
	CPDMatrix Vt;

	void upsampling(const size_t k=200) {
		CPDVector P1inv = 1.0 / P1.array();
		if (!P1inv.allFinite()) replaceInf2ValidNum(P1inv, std::numeric_limits<Prec>::max());

		CPDMatrix Phi, Ene = invSimilarityTransform(P1inv.asDiagonal() * PX) - Y.get();

		if (GT == cpd::GramType::FULL || m < 2000) {
			CPDVector Psy = lambda * variance / (s * s) * P1inv;
			CPDMatrix Gzz = gaussianKernel(Y.get(), beta);
			Gzz += Psy.asDiagonal();
			Phi = (Gzz).llt().solve(CPDMatrix::Identity(m, m)) * Ene;

		} else {
			CPDVector Psyinv = (s * s) / (lambda * variance) * P1;
			Phi = Psyinv.asDiagonal() * (Ene - (Q *
						(Linv.asDiagonal().toDenseMatrix() + QT * Psyinv.asDiagonal() * Q).inverse() * 
						(QT * (Psyinv.asDiagonal() * Ene))));
		}

		CPDMatrix U = randomSampling(vConcatDenseMat(Yorig.get(), Y.get()), k);

		Eigen::JacobiSVD<CPDMatrix> SVD(gaussianKernel(U, beta), Eigen::ComputeThinV);
		auto V = SVD.matrixV();

		CPDVector Guu = SVD.singularValues();
		CPDMatrix Gyu = gaussianKernel(Yorig.get(), U, beta) * (sqrt((Prec)k / Yorig.get().rows()) * V * Guu.asDiagonal().inverse());
		CPDMatrix Guz = (sqrt((Prec)k / m) * V * Guu.asDiagonal().inverse()).transpose() * gaussianKernel(U, Y.get(), beta);
		Guu = (Prec)m * Yorig.get().rows() / (k * k) * Guu;

		Vt = Gyu * (Guu.asDiagonal() * (Guz * Phi));
	}

	public:

		// @param src	: source point set MxD
		// @param trg	: target point set NxD
		// @param ksrc	: sample size of the source point set
		// @param ktrg	: sample size of the target point set
		// @param weight	: amount of noise in the point sets (0.0 <= weight < 1.0)
		// @param gamma	: randomness of the initial variance (0.0 < gamma)
		// @param kgram	: sample size of gram matrix approximation (if kgram == 0, then kgram = sqrt(M))
		// @param lambda	: goodness of maximum likelihood fit and regularization (0.0 < lambda)
		// @param beta	: width of smoothing gaussian filter (0.0 < beta)
		// @param kappa	: randomness of mixing coefficients (0.0 < kappa)
		FastBayesian(Eigen::MatrixBase<Derived1>& src, Eigen::MatrixBase<Derived2>& trg,
									const size_t ksrc, const size_t ktrg,
									const Prec weight = 0.2, const Prec gamma = 1.0,
									size_t kgram = 0, const Prec lambda = 2.0, const Prec beta = 2.0, const Prec kappa = 1.0) :
			DownSampler<Derived1, Derived2, Prec>(src, trg, ksrc, ktrg),
			Bayesian<GT, Derived1, Derived2, Prec>(this->Ysample, this->Xsample, weight, gamma, kgram, lambda, beta, kappa),
			Yorig(src), Xorig(trg), beta(beta), Vt(CPDMatrix::Zero(this->ky, dim)) {}

		// run single iteration
		void update(const bool displayParameters = false) {
			BaseType::update(displayParameters);
		}

		CPDMatrix getTransformedPoints(const size_t k = 100) {
			upsampling(k);
			return similarityTransform(Yorig.get() + Vt);
		}

		// apply transformation input point set
		// @param V : source point set
		// @return transformed point set
		template <typename Derived>
		CPDMatrix transform(const Eigen::MatrixBase<Derived>& V, const size_t k = 100) {
			upsampling(k);
			return similarityTransform(V + Vt);
		}

		// Transform(Y) = (s * (Y + W) * R.transpose()).rowwise() + t.transpose();
		// @return W : displacement matrix MxD, 
		//         R : rotation matrix DxD, 
		//         t : translation vector Dx1, 
		//         s : scaling value 1x1
		std::tuple<CPDMatrix, CPDMatrix, CPDVector, Prec> getTransformations(const size_t k = 100) {
			upsampling(k);
			return std::make_tuple(Vt, this->R, this->t, s);
		}
	};

}