#pragma once

#include <set>
#include <kika/meshio.hpp>
#include <kika/eigutil.hpp>
#include <kika/surface_mesh.hpp>
#include <kika/diff_geometry.hpp>
#include <kika/distance.hpp>
#include <kika/cudautil.hpp>
#include <igl/blue_noise.h>

namespace kika::util {
/// implementation of Fast Approximation of Laplace-Beltrami Eigenproblems [SGP 2018]
class FastApproxEigs {
	kika::heds::SurfaceMesh& mesh;
	kika::diff::DiscreteLaplacian& dsl;

	double surfaceArea;

	cusolverDnHandle_t cusolverHandle;

	void uniformSampling(size_t n, const unsigned int seed = std::random_device{}()) {
		std::mt19937 engine(seed);
		std::vector<uint32_t> idx(mesh.getNumVerts());
		std::iota(idx.begin(), idx.end(), 0);

		std::shuffle(idx.begin(), idx.end(), engine);

		samples.resize(n);
		for (size_t i = 0; i < n; i++) {
			samples[i] = idx.back();
			idx.pop_back();
		}
	}

	void blueNoiseSampling(size_t n) {
		double radius =	sqrt(surfaceArea * sqrt(3.0) / 3.0 / n);

		Eigen::MatrixX3i F = kika::eig::vec2eig<3>(mesh.faceList()).cast<int>();
		Eigen::MatrixX3d B, P;
		Eigen::VectorXi FI;
		igl::blue_noise(mesh.pos, F, radius, B, FI, P);

		std::set<uint32_t> dup;
		samples.reserve(FI.rows());
		for (auto fi : FI) {
			auto v = mesh.faces[fi]->edge->vert->id;
			if (!dup.contains(v)) {
				samples.emplace_back(v);
				dup.insert(v);
			}
		}
	}

	void computeLocalBasis() {
		struct cubic {
			double alpha, beta;
			cubic(double range) :
				alpha(-3. / (range*range)),
				beta(2. / (range*range*range)) {}
			double operator()(double d) { return  alpha * d * d + beta * d * d * d + 1.; }
		};

		struct quintic {
			double alpha, beta, gamma;
			quintic(double range) :
				alpha(-10./ (range*range*range)),
				beta(15. / (range*range*range*range)),
				gamma(-6. / (range*range*range*range*range)) {}
			double operator()(double d) { return alpha*d*d*d + beta*d*d*d*d + gamma*d*d*d*d*d + 1.; }
		};

		struct gauss {
			double alpha;
			gauss(double range) : alpha(log(0.5) / (range * range * 0.2025)) {}
			double operator()(double d) { return exp(alpha * d * d); }
		};

		double range0 =	3. * sqrt(surfaceArea / (samples.size() * std::numbers::pi));
		std::vector<double> ranges(samples.size(), range0);
		double eps = 1e-2;

		auto shrinkRange = [this, eps](auto& dist, auto& range) {
			Eigen::VectorXd xhi;

			while (1) {
				bool flag = true;

				cubic func(range);

				for (size_t j = 0; j < samples.size(); j++) {
					double u = func(dist(samples[j]));
					if (u < eps) {
						flag = false;
						range *= 0.5;
						break;
					}
				}
				if (flag) break;
			}
		};

		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(samples.size());
		std::vector<bool> covered(mesh.getNumVerts(), false);

		kika::gd::VirtualSourceSolver gdsolver(mesh);
		for (size_t i = 0; i < samples.size(); i++) {
			auto dist = gdsolver.solve({ samples[i] }, std::numeric_limits<uint32_t>::max(), ranges[i]).dist();

			//shrinkRange(dist, ranges.back());

			cubic func(ranges[i]);
			for (size_t j = 0; j < mesh.getNumVerts(); j++) {
				if (dist(j) < ranges[i]) {
					covered[j] = true;
					triplets.emplace_back((int)j, (int)i, func(dist(j)));
				}
			}
		}

		for (size_t i = 0; i < mesh.getNumVerts(); i++) {
			if (covered[i]) continue;
			samples.push_back((uint32_t)i);
			ranges.emplace_back(range0);

			auto dist = gdsolver.solve({ samples.back() }, std::numeric_limits<uint32_t>::max(), ranges.back()).dist();

			//shrinkRange(dist, ranges.back());

			cubic func(ranges.back());
			for (size_t j = 0; j < mesh.getNumVerts(); j++) {
				if (dist(j) < ranges.back()) {
					covered[j] = true;
					triplets.emplace_back((int)j, (int)(samples.size()-1), func(dist(j)));
				}
			}
		}

		prolongOp.resize(mesh.getNumVerts(), samples.size());
		prolongOp.setFromTriplets(triplets.begin(), triplets.end());

		Eigen::VectorXd r1 = prolongOp * Eigen::VectorXd::Ones(prolongOp.cols());
		prolongOp = r1.cwiseInverse().asDiagonal() * prolongOp;
	}

	void eigCuda(const Eigen::MatrixXd& L, const Eigen::MatrixXd& M) {
		int n = (int)L.rows();

		kika::cuda::CuData<double> cuA, cuB;

		cuA.alloc(n * n * sizeof(double));
		memcpy(cuA.host, L.data(), cuA.size);
		cuA.host2device();

		cuB.alloc(n * n * sizeof(double));
		memcpy(cuB.host, M.data(), cuB.size);
		cuB.host2device();

		kika::cuda::CuData<double> cuEval;
		cuEval.alloc(n * sizeof(double));

		int lwork{};
		cusolverDnDsygvd_bufferSize(cusolverHandle,
			CUSOLVER_EIG_TYPE_1,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER,
			n, cuA.device, n, cuB.device, n, cuEval.device, &lwork);

		kika::cuda::CuData<int> info;
		info.alloc(sizeof(int));
		kika::cuda::CuData<double> dwork;
		dwork.alloc(sizeof(double) * lwork);

		cusolverDnDsygvd(cusolverHandle,
			CUSOLVER_EIG_TYPE_1,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER,
			n, cuA.device, n, cuB.device, n, cuEval.device, dwork.device, lwork, info.device);
		cudaDeviceSynchronize();

		cuEval.device2host();
		cuA.device2host();

		evals = Eigen::Map<Eigen::VectorXd>(cuEval.host, n);
		evecs = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(cuA.host, n, n, Eigen::OuterStride<>(n));

		cusolverDnDestroy(cusolverHandle);
		cudaDeviceReset();
	}

	void eigCpu(const Eigen::MatrixXd& L, const Eigen::MatrixXd& M) {
		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(L, M, Eigen::ComputeEigenvectors);
		evals = solver.eigenvalues();
		evecs = solver.eigenvectors();
	}

public:
	std::vector<uint32_t> samples;
	Eigen::VectorXd evals;
	Eigen::MatrixXd evecs;
	Eigen::SparseMatrix<double> prolongOp;

	// @param sampleSize : (defalut) 5 * n
	FastApproxEigs(kika::heds::SurfaceMesh& mesh, kika::diff::DiscreteLaplacian& dsl, const size_t n, size_t sampleSize = 0) :
		mesh(mesh), dsl(dsl), cusolverHandle(nullptr) {
		if (sampleSize == 0) sampleSize = std::min<size_t>(5 * n, mesh.getNumVerts());
		if (sampleSize < n) sampleSize = n;
		std::thread setupCuda;
		if (sampleSize > 1000) setupCuda = std::thread([this]() { cusolverDnCreate(&cusolverHandle); });

		dsl.laplacianOp();
		dsl.vertexAreas();

		surfaceArea = mesh.surfaceArea();

		blueNoiseSampling(sampleSize);
		computeLocalBasis();

		Eigen::MatrixXd L = prolongOp.transpose() * dsl.lbo * prolongOp;
		Eigen::MatrixXd M = prolongOp.transpose() * dsl.s0.asDiagonal() * prolongOp;

		if (sampleSize <= 1000) {
			eigCpu(L, M);
		} else {
			if (setupCuda.joinable()) setupCuda.join();
			eigCuda(L, M);
		}

		Eigen::VectorXd m = (evecs.transpose() * M * evecs).diagonal();
		evecs = evecs.array().rowwise() * m.cwiseSqrt().cwiseInverse().array().transpose();
		evecs = evecs.leftCols(n).eval();
		evals = evals.cwiseAbs().topRows(n).eval();
	}

};
}