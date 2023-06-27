#pragma once
#include <kika/vector_field.hpp>

namespace kika::vf {

// (Ref) The Vector Heat Method [Sharp, N. et al. 2020]
class VectorHeatSolver {
	kika::heds::SurfaceMesh& mesh;
	kika::diff::DiscreteLaplacian& dsl;

	double time;

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> lsolver;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Eigen::dcomplex>> csolver;
	bool initPsolver;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> psolver;
public:

	VectorHeatSolver(kika::heds::SurfaceMesh& mesh, kika::diff::DiscreteLaplacian& dsl, const double m = 1.0) : mesh(mesh), dsl(dsl) {
		dsl.laplacianOp();
		dsl.vertexAreas();
		dsl.connectionLaplacianOp();

		time = m * pow(mesh.avgEdgeLength(), 2);

		auto mass = kika::eig::spdiag(dsl.s0);

		lsolver.compute(mass + time * dsl.lbo);
		csolver.compute(mass.cast<Eigen::dcomplex>() + time * dsl.clbo);
		initPsolver = true;
	}

	/// @param delta : indicator vector
	/// @param xhi   : initial direction
	Eigen::VectorXcd compute(const Eigen::VectorXd& delta, const Eigen::VectorXcd& xhi) {
		Eigen::VectorXcd y0 = delta.array() * xhi.array();
		Eigen::VectorXcd y = csolver.solve(y0);

		Eigen::VectorXd u0 = delta.array() * xhi.array().abs();
		Eigen::VectorXd u = lsolver.solve(u0);

		Eigen::VectorXd phi = lsolver.solve(delta);

		Eigen::VectorXcd x =
			u.array()* y.array() / (phi.array() * y.array().abs());

		return x;
	}

	/// @param src : source vertex
	/// @param vec : direction of src
	template <typename T = uint32_t>
	Eigen::VectorXcd solve(const std::vector<T>& src, const std::vector<Eigen::dcomplex>& vec) {
		Eigen::VectorXd delta = Eigen::VectorXd::Zero(mesh.getNumVerts());
		Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(mesh.getNumVerts());
		for (size_t i = 0; i < src.size(); i++) {
			delta[src[i]] = 1.0;
			xhi[src[i]] = vec[i];
		}

		return compute(delta, xhi);
	}

	// global logarithmic map
	// @return uv map complex vector : |V_C| values range [- max geo dist, max geo dist]
	inline Eigen::VectorXcd logarithmicMap(const size_t src, const double initAngle = 0.0) {
		if (initPsolver) {
			psolver.compute(dsl.lbo);
			initPsolver = false;
		}

		size_t nv = mesh.getNumVerts();

		Eigen::VectorXcd h(nv);
		// step. 1
		{
			Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(nv);
			xhi[src] = std::polar(1.0, initAngle);

			Eigen::VectorXcd y = csolver.solve(xhi);
			h = y.array() / y.array().abs();
		}

		auto phi = dsl.oneringAngles();
		Eigen::VectorXcd r(nv);
		// step. 2 
		{
			Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(nv);

			auto heightTriangle = [this](const kika::heds::pte& edge) {
				double area = mesh.area(edge->face);
				double base = mesh.length(edge->next);
				return 2.0 * area / base;
			};

			auto opsCornerAngle = [this](const kika::heds::pte& edge) {
				return mesh.angle(edge->next->pair, edge->next->next);
			};

			for (auto edge : mesh.verts[src]->incidentEdges<kika::heds::Ptr>()) {
				edge = edge->pair;

				if (edge->face != nullptr) {
					double h = heightTriangle(edge->next);
					double theta = opsCornerAngle(edge->next);
					Eigen::dcomplex e(-theta * sin(theta) / (2. * h), (theta * cos(theta) - sin(theta)) / (2. * h));
					xhi[edge->vert->id] += e * std::polar(1.0, phi.coeffRef(edge->vert->id, edge->pair->vert->id));
				}
				if (edge->pair->face != nullptr) {
					double h = heightTriangle(edge->pair);
					double theta = opsCornerAngle(edge->pair->next->next);
					Eigen::dcomplex e(-theta * sin(theta) / (2. * h), -(theta * cos(theta) - sin(theta)) / (2. * h));
					xhi[edge->vert->id] += e * std::polar(1.0, phi.coeffRef(edge->vert->id, edge->pair->vert->id));
				}
				if (edge->face != nullptr) {
					double h = heightTriangle(edge);
					double theta = opsCornerAngle(edge->next);
					double gamma = opsCornerAngle(edge->next->next);
					double alpha = std::numbers::pi / 2. - gamma;
					Eigen::dcomplex e(-(theta * cos(alpha) + cos(alpha - theta) * sin(theta)) / (2.0 * h),
						-(cos(alpha) - cos(alpha - 2 * theta) + 2 * theta * sin(alpha)) / (4. * h));
					xhi[edge->pair->vert->id] += e * std::polar(1.0, phi.coeffRef(edge->pair->vert->id, edge->vert->id));
				}
			}

			Eigen::VectorXcd y = csolver.solve(xhi);

			r = y.array() / y.array().abs();

			r[src] = 0.0;
		}

		Eigen::VectorXd d(nv);
		// (skip step. 3) step. 4
		{
			Eigen::VectorXd div = Eigen::VectorXd::Zero(nv);
			for (const auto& edge : mesh.edges) {
				auto [u, v] = edge->endpoint();
				double l = mesh.length(edge);
				Eigen::dcomplex ruv = std::polar(l, phi.coeff(u, v));
				Eigen::dcomplex rvu = std::polar(l, phi.coeff(v, u));
				Eigen::dcomplex alignRvu = -rvu /ruv;
				Eigen::dcomplex alignRuv = std::conj(alignRvu);
				double w = 0.5 * mesh.cot(edge);
				w += edge->pair->face == nullptr ? 0.0 : 0.5 * mesh.cot(edge->pair);

				div[u] += -w * (std::conj(0.5 * (r[u] + alignRuv * r[v])) * ruv).real();
				div[v] += -w * (std::conj(0.5 * (r[v] + alignRvu * r[u])) * rvu).real();
			}

			d = psolver.solve(div);
			d = d.array() - d[src];
		}

		// step. 3 and 5
		return r.array() / h.array() * d.array();
	}

	//// karcher mean example
	//  std::vector<std::vector<double>> landmarks;
	//  std::vector<uint32_t> lmIndices;
	//  {
	//  	pshape::Sphere primitive(8, 4);
	//  	auto xyz = primitive.xyz.map(pshape::RowMajor);
	//  	auto tri = primitive.tri.map(pshape::RowMajor);
	//  	auto setLandmark = [&landmarks, &lmIndices, &xyz, &tri, &ael](const Eigen::Vector3d& p) {
	//  		for (size_t i = 0, offset = landmarks.size(); i < tri.size(); i++) lmIndices.emplace_back(tri[i] + offset);
	//  
	//  		for (size_t i = 0; i < xyz.size(); i+=3) {
	//  			landmarks.emplace_back(std::vector<double>(3));
	//  			landmarks.back()[0] = 0.5 * ael * xyz[i+0] + p[0];
	//  			landmarks.back()[1] = 0.5 * ael * xyz[i+1] + p[1];
	//  			landmarks.back()[2] = 0.5 * ael * xyz[i+2] + p[2];
	//  		}
	//  	};
	//  
	//  	Eigen::VectorXi input(4);
	//  	input << 684, 746, 675, 1224;
	//  
	//  	size_t src = 0;
	//  
	//  	for (size_t i = 0; i < 100; i++) {
	//  		Eigen::Vector3d p = V.row(src);
	//  		setLandmark(p);
	//  
	//  		auto logMap = gp::dfield::logarithmicMap(mesh, V, N, src);
	//  
	//  		Eigen::Vector2d v = logMap(input, Eigen::all).colwise().mean();
	//  		std::cout << v.transpose() << std::endl;
	//  		Eigen::Vector2d tv = 0.1 * v;
	//  
	//  		Eigen::Index nn;
	//  		(logMap.rowwise() - tv.transpose()).rowwise().squaredNorm().minCoeff(&nn);
	//  
	//  		src = nn;
	//  	}
	//  
	//  	for (auto i : input) {
	//  		setLandmark(V.row(i));
	//  	}
	//  }


};


}