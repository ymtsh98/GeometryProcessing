#pragma once
#include <functional>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <kika/eigutil.hpp>

namespace kika::opt {

#if false
class MinConfPQN {
protected:
	Eigen::MatrixXd N, M;
	double Hdiag;
	Eigen::VectorXd subx, subg;

	virtual double op(const Eigen::VectorXd& x, Eigen::VectorXd& grad) = 0;
	virtual Eigen::VectorXd project(Eigen::VectorXd x) = 0;

	void lbfgsUpdate(const Eigen::VectorXd& y, const Eigen::VectorXd& s, const size_t corrections,
		Eigen::MatrixXd& S, Eigen::MatrixXd& Y, double& Hdiag);

	Eigen::VectorXd lbfgsHvFunc(const Eigen::VectorXd& v);

	double opSubHv(const Eigen::VectorXd& p, Eigen::VectorXd& g);

	Eigen::VectorXd minConfSPG(Eigen::VectorXd x,
		const size_t maxIter = 10, const double progTol = 1e-10, const size_t memory = 10,
		const double suffDec = 1e-4, const double optTol = 1e-6);

	void minConfPQN(Eigen::VectorXd& x, const size_t maxIter = 500, const double progTol = 1e-9,
		const size_t memory = 10, const double suffDec = 1e-4, const double optTol = 1e-5);

	MinConfPQN();

	void freeResources();
};
#endif

class MinConfPQN {
protected:
	Eigen::MatrixXd N, M;
	double Hdiag;
	Eigen::VectorXd subx, subg;

	virtual double op(const Eigen::VectorXd& x, Eigen::VectorXd& grad) = 0;
	virtual Eigen::VectorXd project(Eigen::VectorXd x) = 0;

	void lbfgsUpdate(const Eigen::VectorXd& y, const Eigen::VectorXd& s, const size_t corrections,
		Eigen::MatrixXd& S, Eigen::MatrixXd& Y, double& Hdiag) {
		double yts = (y.transpose() * s).value();

		if (yts > 1e-10) {
			size_t numCorrections = (size_t)S.cols();
			if (numCorrections < corrections) {
				kika::eig::hAppendDenseMat(S, s);
				kika::eig::hAppendDenseMat(Y, y);
			} else {
				S = kika::eig::hConcatDenseMat(S.rightCols(numCorrections-1), s);
				Y = kika::eig::hConcatDenseMat(Y.rightCols(numCorrections-1), y);
			}
			Hdiag = yts / (y.transpose() * y).value();
		}
	}

	Eigen::VectorXd lbfgsHvFunc(const Eigen::VectorXd& v) {
		return v / Hdiag - N * (M.lu().solve(N.transpose() * v));
	}

	double opSubHv(const Eigen::VectorXd& p, Eigen::VectorXd& g) {
		Eigen::VectorXd d = p - subx;
		Eigen::VectorXd Hd = lbfgsHvFunc(d);
		double f = (subg.transpose() * d).value() + 0.5 * (d.transpose() * Hd).value();
		g = subg + Hd;
		return f;
	}

	Eigen::VectorXd minConfSPG(Eigen::VectorXd x,
		const size_t maxIter = 10, const double progTol = 1e-10, const size_t memory = 10,
		const double suffDec = 1e-4, const double optTol = 1e-6) {

		Eigen::VectorXd g;
		double f = this->opSubHv(x, g);
		size_t funcEvals = 1;

		double alpha, t, of;
		Eigen::VectorXd ox, nx, og, ng, ofvals;

		Eigen::VectorXd bx = x;
		double bf = f;

		size_t i = 1;
		while (funcEvals <= maxIter) {
			if (i == 1) {
				alpha = 1.0;
			} else {
				Eigen::VectorXd y = g - og;
				Eigen::VectorXd s = x - ox;

				alpha = ((s.transpose() * s) / (s.transpose() * y)).value();
				if (alpha <= 1e-10 || alpha > 1e+10) alpha = 1.0;
			}

			Eigen::VectorXd d = -alpha * g;
			of = f;
			ox = x;
			og = g;

			d = this->project(x + d) - x;

			double gtd = g.transpose() * d;
			if (gtd > -progTol) break;

			t = (i == 1) ? std::min(1.0, 1.0 / g.array().abs().sum()) : 1.0;

			if (i == 1) {
				ofvals = Eigen::VectorXd::Constant(memory, std::numeric_limits<double>::min());
			}

			if (i <= memory) {
				ofvals[i - 1] = f;
			} else {
				Eigen::VectorXd temp = Eigen::VectorXd::Zero(memory);
				temp.topRows(memory-1) = ofvals.bottomRows(memory-1);
				temp[memory-1] = f;
				ofvals = temp;
			}

			double funcRef = ofvals.maxCoeff();

			nx = x + t * d;

			double nf = this->opSubHv(nx, ng);
			funcEvals += 1;

			while (nf > (funcRef + suffDec* (g.transpose() * (nx - x)).value())) {
				double temp = t;

				double ngtd = ng.transpose() * d;
				double d1 = gtd + ngtd - 3 * (f - nf) / t;
				double d2 = d1*d1 - gtd * ngtd;
				if (d2 < 0.0)
					t /= 2.0;
				else
					t = std::min(t, std::max(0., t - t*(ngtd + sqrt(d2) - d1)/ (ngtd - gtd + 2 *sqrt(d2))));

				if (t < (temp * 1e-3)) {
					t = temp * 1e-3;
				} else if (t >(temp * 0.6)) {
					t = temp * 0.6;
				}

				if ((t * d).array().abs().maxCoeff() < progTol || t == 0.0) {
					t = 0.0;
					nf = f;
					ng = g;
					break;
				}

				nx = x + t * d;

				nf = this->opSubHv(nx, ng);
				funcEvals += 1;
			}

			x = nx;
			f = nf;
			g = ng;

			if (f < bf) {
				bx = x;
				bf = f;
			}

			if (abs(f-of) < progTol) break;
			if ((t * d).array().abs().maxCoeff() < progTol) break;
			if ((this->project(x-g)-x).array().abs().maxCoeff() < optTol) break;

			i++;
		}

		return bx;
	}

	void minConfPQN(Eigen::VectorXd& x, const size_t maxIter = 500, const double progTol = 1e-9,
		const size_t memory = 10, const double suffDec = 1e-4, const double optTol = 1e-5) {
		auto nVars = x.rows();
		x = this->project(x);

		Eigen::VectorXd g;
		double f = this->op(x, g);
		size_t funcEvals = 1;

		if ((project(x-g)-x).array().abs().maxCoeff() < optTol) return;

		size_t iter = 1;
		Eigen::MatrixXd S, Y;
		Eigen::VectorXd p, s, y;
		Eigen::VectorXd nx, ox, ng, og;
		double Hdiag, of;

		while (funcEvals <= maxIter) {
			if (iter == 1) {
				p = this->project(x-g);
				S = Eigen::MatrixXd::Zero(nVars, 0);
				Y = Eigen::MatrixXd::Zero(nVars, 0);
				Hdiag = 1.0;
			} else {
				y = g - og;
				s = x - ox;

				this->lbfgsUpdate(y, s, 10, S, Y, Hdiag);

				auto k = Y.cols();
				Eigen::MatrixXd L = Eigen::MatrixXd::Zero(k, k);
				for (Eigen::Index j = 1; j < k; j++)
					L(Eigen::seq(j, k-1), j-1) = S(Eigen::all, Eigen::seq(j, k-1)).transpose() * Y.col(j-1);

				this->N = kika::eig::hConcatDenseMat(S/Hdiag, Y);
				this->M = kika::eig::hConcatDenseMat(S.transpose()* S/ Hdiag, L);
				kika::eig::vAppendDenseMat(this->M,
					kika::eig::hConcatDenseMat(L.transpose(), -(S.transpose() * Y).diagonal().asDiagonal().toDenseMatrix()));
				this->Hdiag = Hdiag;
				this->subx = x;
				this->subg = g;

				p = minConfSPG(x);
			}

			Eigen::VectorXd d = p - x;
			og = g;
			ox = x;

			double gtd = (g.transpose() * d).value();
			if (gtd > -progTol) break;

			double t = (iter==1) ? std::min(1.0, 1.0/g.array().abs().sum()) : 1.0;

			nx = (t == 1.0) ? p : (x + t * d);

			double nf = this->op(nx, ng);
			funcEvals++;

			of = f;

			while (nf > (f + suffDec * (g.transpose() * (nx - x)).value())) {
				double temp = t;

				double ngtd = ng.transpose() * d;
				double d1 = gtd + ngtd - 3 * (f - nf) / t;
				double d2 = d1*d1 - gtd * ngtd;
				if (d2 < 0.0)
					t /= 2.0;
				else
					t = std::min(t, std::max(0., t - t*(ngtd + sqrt(d2) - d1)/ (ngtd - gtd + 2 *sqrt(d2))));

				if (t < (temp * 1e-3))
					t = temp * 1e-3;
				else if (t >(temp * 0.6))
					t = temp * 0.6;

				if ((t * d).array().abs().maxCoeff() < progTol || t == 0.) {
					t = 0.;
					nf = f;
					ng = g;
					break;
				}

				nx = x + t * d;
				nf = this->op(nx, ng);
				funcEvals++;
			}

			x = nx;
			f = nf;
			g = ng;

			if (abs(f-of) < progTol)	break;
			if ((t*d).array().abs().maxCoeff() < progTol)	break;
			if ((this->project(x-g)-x).array().abs().maxCoeff() < optTol) return;

			iter++;
		}
	}

	MinConfPQN() {}

	void freeResources() {
		N.resize(0, 0);
		M.resize(0, 0);
		subx.resize(0);
		subg.resize(0);
	}
};

#if false
class MinConfPQNCmplx {
protected:
	Eigen::MatrixXcd N, M;
	double Hdiag;
	Eigen::VectorXcd subx, subg;

	virtual double op(const Eigen::VectorXcd& x, Eigen::VectorXcd& grad) = 0;
	virtual Eigen::VectorXcd project(Eigen::VectorXcd x) = 0;

	void lbfgsUpdate(const Eigen::VectorXcd& y, const Eigen::VectorXcd& s, const size_t corrections,
		Eigen::MatrixXcd& S, Eigen::MatrixXcd& Y, double& Hdiag);

	Eigen::VectorXcd lbfgsHvFunc(const Eigen::VectorXcd& v);

	double opSubHv(const Eigen::VectorXcd& p, Eigen::VectorXcd& g);

	Eigen::VectorXcd minConfSPG(Eigen::VectorXcd x,
		const size_t maxIter = 10, const double progTol = 1e-10, const size_t memory = 10,
		const double suffDec = 1e-4, const double optTol = 1e-6);

	void minConfPQN(Eigen::VectorXcd& x, const size_t maxIter = 500, const double progTol = 1e-9,
		const size_t memory = 10, const double suffDec = 1e-4, const double optTol = 1e-5);

	MinConfPQNCmplx();

	void freeResources();
};

class MinConfPQNCmplx {
protected:
	Eigen::MatrixXcd N, M;
	double Hdiag;
	Eigen::VectorXcd subx, subg;

	virtual double op(const Eigen::VectorXcd& x, Eigen::VectorXcd& grad) = 0;
	virtual Eigen::VectorXcd project(Eigen::VectorXcd x) = 0;

	void lbfgsUpdate(const Eigen::VectorXcd& y, const Eigen::VectorXcd& s, const size_t corrections,
		Eigen::MatrixXcd& S, Eigen::MatrixXcd& Y, double& Hdiag) {
		double yts = (y.adjoint() * s).real().value();

		if (yts > 1e-10) {
			size_t numCorrections = (size_t)S.cols();
			if (numCorrections < corrections) {
				kika::eig::hAppendDenseMat(S, s);
				kika::eig::hAppendDenseMat(Y, y);
			} else {
				S = kika::eig::hConcatDenseMat(S.rightCols(numCorrections-1), s);
				Y = kika::eig::hConcatDenseMat(Y.rightCols(numCorrections-1), y);
			}
			Hdiag = yts / (y.adjoint() * y).real().value();
		}
	}

	Eigen::VectorXcd lbfgsHvFunc(const Eigen::VectorXcd& v) {
		return v / Hdiag - N * (M.lu().solve(N.transpose() * v));
	}

	double opSubHv(const Eigen::VectorXcd& p, Eigen::VectorXcd& g) {
		Eigen::VectorXcd d = p - subx;
		Eigen::VectorXcd Hd = lbfgsHvFunc(d);
		double f = (subg.adjoint() * d).real().value() + 0.5 * (d.adjoint() * Hd).real().value();
		g = subg + Hd;
		return f;
	}

	Eigen::VectorXcd minConfSPG(Eigen::VectorXcd x,
		const size_t maxIter = 10, const double progTol = 1e-10, const size_t memory = 10,
		const double suffDec = 1e-4, const double optTol = 1e-6) {

		Eigen::VectorXcd g;
		double f = this->opSubHv(x, g);
		size_t funcEvals = 1;

		double alpha, t, of;
		Eigen::VectorXcd ox, nx, og, ng;
		Eigen::VectorXd ofvals;

		Eigen::VectorXcd bx = x;
		double bf = f;

		size_t i = 1;
		while (funcEvals <= maxIter) {
			if (i == 1) {
				alpha = 1.0;
			} else {
				Eigen::VectorXcd y = g - og;
				Eigen::VectorXcd s = x - ox;

				alpha = ((s.adjoint() * s) / (s.adjoint() * y)).real().value();
				if (alpha <= 1e-10 || alpha > 1e+10) alpha = 1.0;
			}

			Eigen::VectorXcd d = -alpha * g;
			of = f;
			ox = x;
			og = g;

			d = this->project(x + d) - x;

			double gtd = (g.adjoint() * d).real().value();
			if (gtd > -progTol) break;

			t = (i == 1) ? std::min(1.0, 1.0 / g.array().abs2().sum()) : 1.0;

			if (i == 1) {
				ofvals = Eigen::VectorXd::Constant(memory, std::numeric_limits<double>::min());
			}

			if (i <= memory) {
				ofvals[i - 1] = f;
			} else {
				Eigen::VectorXd temp = Eigen::VectorXd::Zero(memory);
				temp.topRows(memory-1) = ofvals.bottomRows(memory-1);
				temp[memory-1] = f;
				ofvals = temp;
			}

			double funcRef = ofvals.maxCoeff();

			nx = x + t * d;

			double nf = this->opSubHv(nx, ng);
			funcEvals += 1;

			while (nf > (funcRef + suffDec* (g.adjoint() * (nx - x)).real().value())) {
				double temp = t;

				double ngtd = (ng.adjoint() * d).real().value();
				double d1 = gtd + ngtd - 3 * (f - nf) / t;
				double d2 = d1*d1 - gtd * ngtd;
				if (d2 < 0.0)
					t /= 2.0;
				else
					t = std::min(t, std::max(0., t - t*(ngtd + sqrt(d2) - d1)/ (ngtd - gtd + 2 *sqrt(d2))));

				if (t < (temp * 1e-3)) {
					t = temp * 1e-3;
				} else if (t >(temp * 0.6)) {
					t = temp * 0.6;
				}

				if ((t * d).array().abs2().maxCoeff() < progTol || t == 0.0) {
					t = 0.0;
					nf = f;
					ng = g;
					break;
				}

				nx = x + t * d;

				nf = this->opSubHv(nx, ng);
				funcEvals += 1;
			}

			x = nx;
			f = nf;
			g = ng;

			if (f < bf) {
				bx = x;
				bf = f;
			}

			if (abs(f-of) < progTol) break;
			if ((t * d).array().abs2().maxCoeff() < progTol) break;
			if ((this->project(x-g)-x).array().abs2().maxCoeff() < optTol) break;

			i++;
		}

		return bx;
	}

	void minConfPQN(Eigen::VectorXcd& x, const size_t maxIter = 500, const double progTol = 1e-9,
		const size_t memory = 10, const double suffDec = 1e-4, const double optTol = 1e-5) {
		auto nVars = x.rows();
		x = this->project(x);

		Eigen::VectorXcd g;
		double f = this->op(x, g);
		size_t funcEvals = 1;

		if ((project(x-g)-x).array().abs2().maxCoeff() < optTol) return;

		size_t iter = 1;
		Eigen::MatrixXcd S, Y;
		Eigen::VectorXcd p, s, y;
		Eigen::VectorXcd nx, ox, ng, og;
		double Hdiag, of;

		while (funcEvals <= maxIter) {
			if (iter == 1) {
				p = this->project(x-g);
				S = Eigen::MatrixXcd::Zero(nVars, 0);
				Y = Eigen::MatrixXcd::Zero(nVars, 0);
				Hdiag = 1.0;
			} else {
				y = g - og;
				s = x - ox;

				this->lbfgsUpdate(y, s, 10, S, Y, Hdiag);

				auto k = Y.cols();
				Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(k, k);
				for (Eigen::Index j = 1; j < k; j++)
					L(Eigen::seq(j, k-1), j-1) = S(Eigen::all, Eigen::seq(j, k-1)).adjoint() * Y.col(j-1);

				this->N = kika::eig::hConcatDenseMat(S/Hdiag, Y);
				this->M = kika::eig::hConcatDenseMat(S.transpose()* S/ Hdiag, L);
				kika::eig::vAppendDenseMat(this->M,
					kika::eig::hConcatDenseMat(L.transpose(), -(S.transpose() * Y).diagonal().asDiagonal().toDenseMatrix()));
				this->Hdiag = Hdiag;
				this->subx = x;
				this->subg = g;

				p = minConfSPG(x);
			}

			Eigen::VectorXcd d = p - x;
			og = g;
			ox = x;

			double gtd = (g.adjoint() * d).real().value();
			if (gtd > -progTol) break;

			double t = (iter==1) ? std::min(1.0, 1.0/g.array().abs2().sum()) : 1.0;

			nx = (t == 1.0) ? p : (x + t * d);

			double nf = this->op(nx, ng);
			funcEvals++;

			of = f;

			while (nf > (f + suffDec * (g.adjoint() * (nx - x)).real().value())) {
				double temp = t;

				double ngtd = (ng.adjoint() * d).real().value();
				double d1 = gtd + ngtd - 3 * (f - nf) / t;
				double d2 = d1*d1 - gtd * ngtd;
				if (d2 < 0.0)
					t /= 2.0;
				else
					t = std::min(t, std::max(0., t - t*(ngtd + sqrt(d2) - d1)/ (ngtd - gtd + 2 *sqrt(d2))));

				if (t < (temp * 1e-3))
					t = temp * 1e-3;
				else if (t >(temp * 0.6))
					t = temp * 0.6;

				if ((t * d).array().abs2().maxCoeff() < progTol || t == 0.) {
					t = 0.;
					nf = f;
					ng = g;
					break;
				}

				nx = x + t * d;
				nf = this->op(nx, ng);
				funcEvals++;
			}

			x = nx;
			f = nf;
			g = ng;

			if (abs(f-of) < progTol)	break;
			if ((t*d).array().abs2().maxCoeff() < progTol)	break;
			if ((this->project(x-g)-x).array().abs2().maxCoeff() < optTol) return;

			iter++;
		}
	}

	MinConfPQNCmplx() {}

	void freeResources() {
		N.resize(0, 0);
		M.resize(0, 0);
		subx.resize(0);
		subg.resize(0);
	}
};
#endif

}