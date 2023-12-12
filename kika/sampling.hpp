#pragma once
#include <kika/eigutil.hpp>
#include <kika/surface_mesh.hpp>
#include <kika/distance.hpp>
#include <set>

#pragma warning(push)
#pragma warning(disable:4018 4244 4267 4819)
#include <igl/blue_noise.h>
#pragma warning(pop)

#if false
namespace kika::samp {

class BlueNoiseSampler {
	kika::heds::SurfaceMesh& mesh;
	double surfaceArea;
	std::vector<double> ranges;

public:
	std::vector<uint32_t> samples;
	Eigen::SparseMatrix<double> prolongOp;

	//// 
	//// template <typename T>
	//// void sampling(size_t n, const std::vector<T>& init = {}) {
	//// 	double radius = sqrt(surfaceArea * sqrt(3.0) / 3.0 / n);
	//// 
	//// 	Eigen::MatrixX3i F = kika::eig::vec2eig<3>(mesh.faceList()).cast<int>();
	//// 	Eigen::MatrixX3d B, P;
	//// 	Eigen::VectorXi FI;
	//// 
	//// 	#pragma warning(push)
	//// 	#pragma warning(disable:4018 4244 4267)
	//// 	igl::blue_noise(mesh.pos, F, radius, B, FI, P);
	//// 	#pragma warning(pop)
	//// 
	//// 	std::set<uint32_t> dup;
	//// 	samples.reserve(FI.rows());
	//// 
	//// 	for (size_t i = 0; i < init.size(); i++) {
	//// 		if (!dup.contains(static_cast<uint32_t>(init[i]))) {
	//// 			samples.emplace_back(static_cast<uint32_t>(init[i]));
	//// 			dup.insert(static_cast<uint32_t>(init[i]));
	//// 		}
	//// 	}
	//// 
	//// 	for (auto fi : FI) {
	//// 		Eigen::Index minIndex = 0;
	//// 		Eigen::Vector3d c = mesh.center(mesh.faces[fi]);
	//// 		auto&& verts = mesh.faces[fi]->circuitVerts();
	//// 		(mesh.pos(verts, Eigen::all).rowwise() - c.transpose()).rowwise().squaredNorm().minCoeff(&minIndex);
	//// 
	//// 		auto v = (uint32_t)verts[minIndex];
	//// 		if (!dup.contains(v)) {
	//// 			samples.emplace_back(v);
	//// 			dup.insert(v);
	//// 		}
	//// 	}
	//// }
	//// 
	//// void computeLocalBasis() {
	//// 	struct cubic {
	//// 		double alpha, beta;
	//// 		cubic(double range) :
	//// 			alpha(-3. / (range * range)),
	//// 			beta(2. / (range * range * range)) {}
	//// 		double operator()(double d) { return  alpha * d * d + beta * d * d * d + 1.; }
	//// 	};
	//// 
	//// 	struct quintic {
	//// 		double alpha, beta, gamma;
	//// 		quintic(double range) :
	//// 			alpha(-10. / (range * range * range)),
	//// 			beta(15. / (range * range * range * range)),
	//// 			gamma(-6. / (range * range * range * range * range)) {}
	//// 		double operator()(double d) { return alpha * d * d * d + beta * d * d * d * d + gamma * d * d * d * d * d + 1.; }
	//// 	};
	//// 
	//// 	struct gauss {
	//// 		double alpha;
	//// 		gauss(double range) : alpha(log(0.5) / (range * range * 0.2025)) {}
	//// 		double operator()(double d) { return exp(alpha * d * d); }
	//// 	};
	//// 
	//// 	double range0 = 3. * sqrt(surfaceArea / (samples.size() * std::numbers::pi));
	//// 	ranges.resize(samples.size(), range0);
	//// 	double eps = 1e-2;
	//// 
	//// 	std::vector<Eigen::Triplet<double>> triplets;
	//// 	std::vector<bool> covered(mesh.getNumVerts(), false);
	//// 
	//// 	kika::gd::VirtualSourceSolver gdsolver(mesh);
	//// 	for (size_t i = 0; i < samples.size(); i++) {
	//// 		auto dist = gdsolver.solve({ samples[i] }, std::numeric_limits<uint32_t>::max(), ranges[i]).dist();
	//// 
	//// 		cubic func(ranges[i]);
	//// 		for (size_t j = 0; j < mesh.getNumVerts(); j++) {
	//// 			if (dist(j) < ranges[i]) {
	//// 				covered[j] = true;
	//// 				triplets.emplace_back((int)j, (int)i, func(dist(j)));
	//// 			}
	//// 		}
	//// 	}
	//// 
	//// 	for (size_t i = 0; i < mesh.getNumVerts(); i++) {
	//// 		if (covered[i]) continue;
	//// 		samples.push_back((uint32_t)i);
	//// 		ranges.emplace_back(range0);
	//// 
	//// 		auto dist = gdsolver.solve({ samples.back() }, std::numeric_limits<uint32_t>::max(), ranges.back()).dist();
	//// 
	//// 		cubic func(ranges.back());
	//// 		for (size_t j = 0; j < mesh.getNumVerts(); j++) {
	//// 			if (dist(j) < ranges.back()) {
	//// 				covered[j] = true;
	//// 				triplets.emplace_back((int)j, (int)(samples.size() - 1), func(dist(j)));
	//// 			}
	//// 		}
	//// 	}
	//// 
	//// 	prolongOp.resize(mesh.getNumVerts(), samples.size());
	//// 	prolongOp.setFromTriplets(triplets.begin(), triplets.end());
	//// 
	//// 	Eigen::VectorXd r1 = prolongOp * Eigen::VectorXd::Ones(prolongOp.cols());
	//// 	
	//// 	prolongOp = r1.cwiseInverse().asDiagonal() * prolongOp;
	//// }

	void build(size_t n) {
		struct cubic {
			double alpha, beta;
			cubic(double range) :
				alpha(-3. / (range * range)),
				beta(2. / (range * range * range)) {}
			double operator()(double d) { return  alpha * d * d + beta * d * d * d + 1.; }
		};

		size_t nv = mesh.getNumVerts();
		double surfaceArea = mesh.surfaceArea();
		double rangeS = sqrt(surfaceArea * sqrt(3.0) / 3.0 / n); // sample range
		double rangeP = 3. * sqrt(surfaceArea / (n * std::numbers::pi)); // prolong range

		auto order = kika::util::randomPermutation<uint32_t>(nv);

		std::vector<uint32_t> arrived(nv, 0);
		samples.reserve(n * 2);
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(n * 2);

		kika::gd::VirtualSourceSolver gdsolver(mesh);
		cubic func(rangeP);

		for (size_t i = 0; i < samples.size(); i++) {
			auto s = samples[i];
			arrived[s]++;
			auto& dist = gdsolver.solve({ s }, -1, rangeP).dist();
			for (size_t j = 0; j < nv; j++) {
				arrived[j] += (dist[j] < rangeS);
				if (dist[j] < rangeP) {
					triplets.emplace_back((int)j, (int)i, func(dist(j)));
				}
			}
		}


		for (size_t i = 0; i < nv; i++) {
			auto s = order[i];
			if (arrived[s] != 0) continue;

			auto& dist = gdsolver.solve({ s }, -1, rangeP).dist();
			for (size_t j = 0; j < nv; j++) {
				arrived[j] += (dist[j] < rangeS);
				if (dist[j] < rangeP) {
					triplets.emplace_back((int)j, (int)samples.size(), func(dist(j)));
				}
			}
			samples.emplace_back(s);
		}

		prolongOp.resize(mesh.getNumVerts(), samples.size());
		prolongOp.setFromTriplets(triplets.begin(), triplets.end());
		Eigen::VectorXd r1 = prolongOp * Eigen::VectorXd::Ones(prolongOp.cols());
		prolongOp = r1.cwiseInverse().asDiagonal() * prolongOp;

	}

	BlueNoiseSampler(kika::heds::SurfaceMesh& mesh, size_t sampleSize) :	mesh(mesh) {
		if (sampleSize == 0 || sampleSize > mesh.getNumVerts()) {
			samples.resize(mesh.getNumVerts());
			std::iota(samples.begin(), samples.end(), 0);
			prolongOp.resize(mesh.getNumVerts(), mesh.getNumVerts());
			prolongOp.setIdentity();
		} else {
			surfaceArea = mesh.surfaceArea();
			build(sampleSize);
			// sampling<uint32_t>(sampleSize);
			// computeLocalBasis();
		}
	}

	template <typename T>
	BlueNoiseSampler(kika::heds::SurfaceMesh& mesh, size_t sampleSize, const std::vector<T>& init) : mesh(mesh) {
		if (sampleSize == 0 || sampleSize >= mesh.getNumVerts()) {
			samples.resize(mesh.getNumVerts());
			std::iota(samples.begin(), samples.end(), 0);
			prolongOp.resize(mesh.getNumVerts(), mesh.getNumVerts());
			prolongOp.setIdentity();
		} else {
			surfaceArea = mesh.surfaceArea();
			sampleSize = (size_t)std::max<int>(0, ((int)sampleSize - init.size()));

			std::set<uint32_t> dup;
			samples.reserve(init.size());
			
			for (size_t i = 0; i < init.size(); i++) {
				if (!dup.contains(static_cast<uint32_t>(init[i]))) {
					samples.emplace_back(static_cast<uint32_t>(init[i]));
					dup.insert(static_cast<uint32_t>(init[i]));
				}
			}
			
			build(sampleSize);
			//sampling(sampleSize, init);
			//computeLocalBasis();
		}
	}
};

}
#endif

namespace kika::samp {

struct CubicFilter {
  double alpha, beta;
  CubicFilter(double range) :
    alpha(-3. / (range * range)),
    beta(2. / (range * range * range)) {}
  double operator()(double d) { return  alpha * d * d + beta * d * d * d + 1.; }
};

struct QuinticFilter {
  double alpha, beta, gamma;
  QuinticFilter(double range) :
    alpha(-10. / (range * range * range)),
    beta(15. / (range * range * range * range)),
    gamma(-6. / (range * range * range * range * range)) {}
  double operator()(double d) { return alpha * d * d * d + beta * d * d * d * d + gamma * d * d * d * d * d + 1.; }
};

struct GaussFilter {
  double alpha;
  GaussFilter(double range) : alpha(-range) //log(0.5) / (range * range * 0.2025)
  {}
  double operator()(double d) { return exp(alpha * d * d); }
};

struct NonFilter {
  NonFilter(double) {}
  double operator()(double d) { return  d; }
};

struct SquareFilter {
  SquareFilter(double) {}
  double operator()(double d) { return  d * d; }
};

class BlueNoiseSampler {

  const heds::SurfaceMesh* mesh;
  double surfaceArea;

  std::vector<uint32_t> presamples;
public:
  std::vector<uint32_t> samples;
  Eigen::SparseMatrix<double> prolongOp;

  BlueNoiseSampler(const heds::SurfaceMesh& mesh) : mesh(&mesh), surfaceArea(mesh.surfaceArea()), presamples({}), samples({}) {}

  template <typename T = uint32_t>
  BlueNoiseSampler(const heds::SurfaceMesh& mesh, const std::vector<T>& init) : mesh(&mesh), surfaceArea(mesh.surfaceArea()), presamples({}), samples({}) {
    std::set<uint32_t> dups;
    for (const auto& value : init) dups.insert(static_cast<uint32_t>(init));

    presamples.reserve(dups.size());
    for (const auto& value : dups) presamples.emplace_back(value);
  }

  std::vector<uint32_t>& sampling(const size_t n, const uint32_t seed = std::random_device{}()) {
    if (mesh == nullptr) return samples;

    if (n == 0 || n > mesh->getNumVerts()) {
      samples.resize(mesh->getNumVerts());
      std::iota(samples.begin(), samples.end(), 0);
      return samples;
    }

    // (rough approximation) set sample size to n
    double range = sqrt(surfaceArea * sqrt(3.0) / 3.0 / n);

    Eigen::MatrixX3i F = eig::vec2eig<3>(mesh->faceList()).cast<int>();
    Eigen::MatrixX3d B, P;
    Eigen::VectorXi FI;

    igl::blue_noise(mesh->pos, F, range, B, FI, P);

    samples.clear();
    samples.reserve(FI.rows() + presamples.size());

    // init values
    std::set<uint32_t> dups(presamples.begin(), presamples.end());
    for (size_t i = 0; i < presamples.size(); i++) samples.emplace_back(static_cast<uint32_t>(presamples[i]));

    for (const auto& fi : FI) {
      Eigen::Index minIndex = 0;
      Eigen::Vector3d c = mesh->center(mesh->faces[fi]);
      auto&& verts = mesh->faces[fi]->circuitVerts();
      (mesh->pos(verts, Eigen::all).rowwise() - c.transpose()).rowwise().squaredNorm().minCoeff(&minIndex);

      auto v = (uint32_t)verts[minIndex];
      if (!dups.contains(v)) {
        samples.emplace_back(v);
        dups.insert(v);
      }
    }

    return samples;
  }

  // WARNINGS : prolong maybe extends [samples].
  template <typename GeoDistSolver = kika::gd::VirtualSourceSolver, typename Filter = CubicFilter, bool Normalize = true>
  Eigen::SparseMatrix<double>& prolong() {
    if (mesh == nullptr || samples.size() == 0) return prolongOp;

    size_t nv = mesh->getNumVerts();

    if (samples.size() >= nv) {
      prolongOp = eig::speye(nv);
      return prolongOp;
    }

    double radius = 3. * sqrt(surfaceArea / (samples.size() * std::numbers::pi));

    std::vector<bool> covered(nv, false);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nv * 100);

    GeoDistSolver gdsolver(*mesh);

    Filter func(radius);
    // #pragma omp parallel for firstprivate(gdsolver)
    for (int i = 0; i < samples.size(); i++) {
      auto&& dist = gdsolver.solve({ samples[i] }, std::numeric_limits<uint32_t>::max(), radius).dist();

      for (size_t j = 0; j < nv; j++) {
        if (dist(j) < radius) {
          covered[j] = true;
          triplets.emplace_back((int)j, (int)i, func(dist(j)));
        }
      }
    }

    // additinal sampling for uncovered vertices
    for (size_t i = 0; i < nv; i++) {
      if (covered[i]) continue;
      samples.push_back((uint32_t)i);

      auto dist = gdsolver.solve({ samples.back() }, std::numeric_limits<uint32_t>::max(), radius).dist();

      for (size_t j = 0; j < nv; j++) {
        if (dist(j) < radius) {
          covered[j] = true;
          triplets.emplace_back((int)j, (int)(samples.size() - 1), func(dist(j)));
        }
      }
    }

    prolongOp.resize(nv, samples.size());
    prolongOp.setFromTriplets(triplets.begin(), triplets.end());

    if (Normalize) prolongOp = kika::eig::sumNormalized(prolongOp);

    return prolongOp;
  }

  // it's not fast?
  #if false
  std::vector<uint32_t>& samplingNprolong(size_t n, const uint32_t seed = std::random_device{}()) {
    samples.clear();
    if (mesh == nullptr) return samples;
    n = clamp(n);

    size_t nv = mesh->getNumVerts();

    // (rough approximation) set sample size to n, search range
    double radius = 3. * sqrt(surfaceArea / (n * std::numbers::pi));
    double range = sqrt(surfaceArea * sqrt(3.0) / 3.0 / n);

    auto order = util::randomPermutation<uint32_t>(nv, seed);

    samples.reserve(n * 2);

    std::vector<uint32_t> arrived(nv, 0);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nv * 100);

    gd::VirtualSourceSolver gdsolver(*mesh);
    cubic func(radius);

    // #pragma omp parallel for firstprivate(gdsolver)
    for (int i = 0; i < nv; i++) {
      auto s = order[i];
      if (arrived[s] != 0) continue;

      auto& dist = gdsolver.solve({ s }, -1, radius).dist();

      //#pragma omp critical
      {

        for (int j = 0; j < nv; j++) {
          arrived[j] += (dist[j] < range);
          if (dist[j] < radius) {
            triplets.emplace_back((int)j, (int)samples.size(), dist(j));
          }
        }
        samples.emplace_back(s);
      }
    }

    Eigen::SparseMatrix<double> op(nv, samples.size());
    op.setFromTriplets(triplets.begin(), triplets.end());
    op.unaryExpr(func);
    eig::sumNormalized(op);

    return samples;
  }
  #endif

};

// base class
#if false
class VertexSampler {
protected:
  struct cubic {
    double alpha, beta;
    cubic(double range) :
      alpha(-3. / (range * range)),
      beta(2. / (range * range * range)) {}
    double operator()(double d) { return  alpha * d * d + beta * d * d * d + 1.; }
  };

  struct quintic {
    double alpha, beta, gamma;
    quintic(double range) :
      alpha(-10. / (range * range * range)),
      beta(15. / (range * range * range * range)),
      gamma(-6. / (range * range * range * range * range)) {}
    double operator()(double d) { return alpha * d * d * d + beta * d * d * d * d + gamma * d * d * d * d * d + 1.; }
  };

  struct gauss {
    double alpha;
    gauss(double range) : alpha(log(0.5) / (range * range * 0.2025)) {}
    double operator()(double d) { return exp(-alpha * d * d); }
  };

  const size_t clamp(const size_t n) {
    if (n == 0 || n > mesh->getNumVerts()) return mesh->getNumVerts();
    return n;
  }

  const heds::SurfaceMesh* mesh;

public:
  std::vector<uint32_t> samples;

  VertexSampler() : mesh(nullptr), samples({}) {}
  VertexSampler(const heds::SurfaceMesh& mesh) : mesh(&mesh) {}
  VertexSampler(const heds::SurfaceMesh& mesh, const size_t n) : mesh(nullptr) {
    sampling(n);
  }

  std::vector<uint32_t>& build(const heds::SurfaceMesh& mesh, const size_t n) {
    this->mesh = &mesh;
    return sampling(n);
  }

  virtual std::vector<uint32_t>& sampling(size_t n, const uint32_t seed = std::random_device{}()) {
    samples.clear();

    if (mesh == nullptr) return samples;
    n = clamp(n);

    samples.resize(n);
    std::iota(samples.begin(), samples.begin() + n, 0);

    return samples;
  }

};

class UniformSampler : public VertexSampler {
public:
  UniformSampler() : VertexSampler() {}
  UniformSampler(const heds::SurfaceMesh& mesh) : VertexSampler(mesh) {}
  UniformSampler(const heds::SurfaceMesh& mesh, const size_t n) : VertexSampler(mesh) {
    sampling(n);
  }

  std::vector<uint32_t>& sampling(size_t n, const uint32_t seed = std::random_device{}()) final {
    samples.clear();
    if (mesh == nullptr) return samples;
    n = clamp(n);

    std::mt19937 engine(seed);

    std::vector<uint32_t> idx(mesh->getNumVerts());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), engine);

    std::memcpy(samples.data(), idx.data(), n * sizeof(uint32_t));

    return samples;
  }
};
#endif

}