#pragma once
#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>

namespace gp::field {
/// geodesic heat vector
/// 
/// usage : 
/// auto x = vectorHeat(...)
/// Eigen::MatrixX3d vfields(nv, 3);
/// for (size_t i = 0; i < nv; i++) {
/// 	Eigen::Vector3d n = N.row(i).normalized();
/// 	Eigen::Vector3d e =	(V.row(mesh.verts[i]->edge->pair->vert->id) - V.row(i)).normalized();
/// 	vfields.row(i) = quatRot(e, n, std::arg(x[i]));
/// }
/// 
/// @param delta : indicator vector
/// @param xhi   : initial direction
/// @param mesh  
inline Eigen::VectorXcd vectorHeat(
	const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, 
	const Eigen::VectorXd& delta, const Eigen::VectorXcd& xhi) {
	double ael = mesh.avgEdgeLength();
	auto lbo = gp::lbo::cotanLaplacian(mesh);
	auto areas = gp::lbo::voronoiAreas(mesh);
	auto clbo = gp::lbo::connectionLaplacian(mesh);

	size_t nv = mesh.getNumVerts();
	double t = ael * ael;
	auto mass = gp::eig::spIdentity<double>(nv);
	mass.diagonal().array() *= areas.array();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> lsolver(mass + t * lbo);
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Eigen::dcomplex>> csolver(mass.cast<Eigen::dcomplex>() + t * clbo);

	Eigen::VectorXcd y0 = delta.array() * xhi.array();
	Eigen::VectorXcd y = csolver.solve(y0);

	Eigen::VectorXd u0 = delta.array() * xhi.unaryExpr([](auto& value) { return std::norm(value); }).array();
	Eigen::VectorXd u = lsolver.solve(u0);

	Eigen::VectorXd phi = lsolver.solve(delta);

	Eigen::VectorXcd x = u.array()* y.array() /
		(phi.array() * y.unaryExpr([](auto& value) { return std::norm(value); }).array());

	return x;
}

// global logarithmic map
// @return uv map : [-1.0, 1.0]
inline Eigen::MatrixX2d logarithmicMap(const heds::Mesh<Eigen::MatrixX3d>& mesh, const size_t src, const double initAngle = 0.0) {
	size_t nv = mesh.getNumVerts();
	double ael = mesh.avgEdgeLength();
	double t = ael * ael;
	auto lbo = gp::lbo::cotanLaplacian(mesh);
	auto clbo = gp::lbo::connectionLaplacian(mesh);
	auto areas = gp::lbo::voronoiAreas(mesh);
	auto mass = gp::eig::spIdentity<double>(nv);
	mass.diagonal().array() *= areas.array();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> psolver(lbo);
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Eigen::dcomplex>> csolver(mass.cast<Eigen::dcomplex>() + t * clbo);

	auto phi = gp::lbo::oneringAngles(mesh);

	Eigen::VectorXcd h(nv);
	// step. 1
	{
		Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(nv);
		xhi[src] = std::polar(1.0, initAngle);

		Eigen::VectorXcd y = csolver.solve(xhi);
		h = y.array() / y.array().abs();// y.unaryExpr([](auto& value) { return std::norm(value); }).array();
	}

	Eigen::VectorXcd r(nv);
	// step. 2 
	{
		Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(nv);

		auto heightTriangle = [&mesh](const gp::heds::pte& edge) {
			double area = mesh.area(edge->face);
			double base = mesh.length(edge->next);
			return 2.0 * area / base;
		};

		auto opsCornerAngle = [&mesh](const gp::heds::pte& edge) {
			auto v = mesh.position(edge->vert->id);
			auto u = mesh.position(edge->next->vert->id);
			auto w = mesh.position(edge->next->next->vert->id);
			return gp::util::angle(v - u, w - u);
		};

		for (auto edge : mesh.verts[src]->incidentEdges()) {
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

		r = y.array() / y.array().abs();// y.unaryExpr([](auto& value) { return std::norm(value); }).array();

		r[src] = 0.0;
	}

	Eigen::VectorXd d(nv);
	// (skip step. 3) step. 4
	{
		//auto complexDot = [](const auto& l, const auto& r) ->double
		//{ return l.real() * r.real() + l.imag() * r.imag(); };

		Eigen::VectorXd div = Eigen::VectorXd::Zero(nv);
		for (const auto& edge : mesh.edges) {
			auto [u, v] = edge->endpointIndex();
			double l = mesh.length(edge);
			Eigen::dcomplex ruv = std::polar(l, phi.coeff(u, v));
			Eigen::dcomplex rvu = std::polar(l, phi.coeff(v, u));
			Eigen::dcomplex alignRvu = -rvu /ruv;
			Eigen::dcomplex alignRuv = std::conj(alignRvu);
			double w = 0.5 * mesh.cot(edge);
			w += edge->pair->face == nullptr ? 0.0 : 0.5 * mesh.cot(edge->pair);

			//div[u] += -w * complexDot(0.5 * (r[u] + alignRuv * r[v]), ruv);
			//div[v] += -w * complexDot(0.5 * (r[v] + alignRvu * r[u]), rvu);
			div[u] += -w * (std::conj(0.5 * (r[u] + alignRuv * r[v])) * ruv).real();
			div[v] += -w * (std::conj(0.5 * (r[v] + alignRvu * r[u])) * rvu).real();
		}

		d = psolver.solve(div);
		d = d.array() - d[src];
	}

	// step. 3 and 5
	Eigen::MatrixX2d logMap(nv, 2);
	for (size_t i = 0; i < nv; i++) {
		Eigen::dcomplex theta = (r[i] / h[i]) * d[i];
		logMap.row(i) << theta.real(), theta.imag();
	}

	return logMap;
}

// global logarithmic map
// @return uv map : [-1.0, 1.0]
inline std::pair<Eigen::MatrixXcd, Eigen::VectorXd> logarithmicMapCmplx(const heds::Mesh<Eigen::MatrixX3d>& mesh, const size_t src, const double initAngle = 0.0) {
	size_t nv = mesh.getNumVerts();
	double ael = mesh.avgEdgeLength();
	double t = ael * ael;
	auto lbo = gp::lbo::cotanLaplacian(mesh);
	auto clbo = gp::lbo::connectionLaplacian(mesh);
	auto areas = gp::lbo::voronoiAreas(mesh);
	auto mass = gp::eig::spIdentity<double>(nv);
	mass.diagonal().array() *= areas.array();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> psolver(lbo);
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Eigen::dcomplex>> csolver(mass.cast<Eigen::dcomplex>() + t * clbo);

	auto phi = gp::lbo::oneringAngles(mesh);

	Eigen::VectorXcd h(nv);
	// step. 1
	{
		Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(nv);
		xhi[src] = std::polar(1.0, initAngle);

		Eigen::VectorXcd y = csolver.solve(xhi);
		h = y.array() / y.array().abs();// y.unaryExpr([](auto& value) { return std::norm(value); }).array();
	}

	Eigen::VectorXcd r(nv);
	// step. 2 
	{
		Eigen::VectorXcd xhi = Eigen::VectorXcd::Zero(nv);

		auto heightTriangle = [&mesh](const gp::heds::pte& edge) {
			double area = mesh.area(edge->face);
			double base = mesh.length(edge->next);
			return 2.0 * area / base;
		};

		auto opsCornerAngle = [&mesh](const gp::heds::pte& edge) {
			auto v = mesh.position(edge->vert->id);
			auto u = mesh.position(edge->next->vert->id);
			auto w = mesh.position(edge->next->next->vert->id);
			return gp::util::angle(v - u, w - u);
		};

		for (auto edge : mesh.verts[src]->incidentEdges()) {
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

		r = y.array() / y.array().abs();// y.unaryExpr([](auto& value) { return std::norm(value); }).array();

		r[src] = 0.0;
	}

	Eigen::VectorXd d(nv);
	// (skip step. 3) step. 4
	{
		//auto complexDot = [](const auto& l, const auto& r) ->double
		//{ return l.real() * r.real() + l.imag() * r.imag(); };

		Eigen::VectorXd div = Eigen::VectorXd::Zero(nv);
		for (const auto& edge : mesh.edges) {
			auto [u, v] = edge->endpointIndex();
			double l = mesh.length(edge);
			Eigen::dcomplex ruv = std::polar(l, phi.coeff(u, v));
			Eigen::dcomplex rvu = std::polar(l, phi.coeff(v, u));
			Eigen::dcomplex alignRvu = -rvu /ruv;
			Eigen::dcomplex alignRuv = std::conj(alignRvu);
			double w = 0.5 * mesh.cot(edge);
			w += edge->pair->face == nullptr ? 0.0 : 0.5 * mesh.cot(edge->pair);

			//div[u] += -w * complexDot(0.5 * (r[u] + alignRuv * r[v]), ruv);
			//div[v] += -w * complexDot(0.5 * (r[v] + alignRvu * r[u]), rvu);
			div[u] += -w * (std::conj(0.5 * (r[u] + alignRuv * r[v])) * ruv).real();
			div[v] += -w * (std::conj(0.5 * (r[v] + alignRvu * r[u])) * rvu).real();
		}

		d = psolver.solve(div);
		d = d.array() - d[src];
	}

	// step. 3 and 5
	Eigen::MatrixXcd logMap(nv, 2);
	logMap.col(0) = r;
	logMap.col(1) = h;
	//for (size_t i = 0; i < nv; i++) {
	//	Eigen::dcomplex theta = (r[i]);// / h[i]);
	//	logMap[i] = theta;
	//}

	return std::make_pair(logMap, d);
}

//// karcher mean example
////
////std::vector<std::vector<double>> landmarks;
////std::vector<uint32_t> lmIndices;
////{
////	pshape::Sphere primitive(8, 4);
////	auto xyz = primitive.xyz.map(pshape::RowMajor);
////	auto tri = primitive.tri.map(pshape::RowMajor);
////	auto setLandmark = [&landmarks, &lmIndices, &xyz, &tri, &ael](const Eigen::Vector3d& p) {
////		for (size_t i = 0, offset = landmarks.size(); i < tri.size(); i++) lmIndices.emplace_back(tri[i] + offset);
////
////		for (size_t i = 0; i < xyz.size(); i+=3) {
////			landmarks.emplace_back(std::vector<double>(3));
////			landmarks.back()[0] = 0.5 * ael * xyz[i+0] + p[0];
////			landmarks.back()[1] = 0.5 * ael * xyz[i+1] + p[1];
////			landmarks.back()[2] = 0.5 * ael * xyz[i+2] + p[2];
////		}
////	};
////
////	Eigen::VectorXi input(4);
////	input << 684, 746, 675, 1224;
////
////	size_t src = 0;
////
////	for (size_t i = 0; i < 100; i++) {
////		Eigen::Vector3d p = V.row(src);
////		setLandmark(p);
////
////		auto logMap = gp::dfield::logarithmicMap(mesh, V, N, src);
////
////		Eigen::Vector2d v = logMap(input, Eigen::all).colwise().mean();
////		std::cout << v.transpose() << std::endl;
////		Eigen::Vector2d tv = 0.1 * v;
////
////		Eigen::Index nn;
////		(logMap.rowwise() - tv.transpose()).rowwise().squaredNorm().minCoeff(&nn);
////
////		src = nn;
////	}
////
////	for (auto i : input) {
////		setLandmark(V.row(i));
////	}
////}




}

namespace gp::field {
// convert intrinsic vector field to extrinsic.
inline Eigen::MatrixX3d cvt2extrinsic(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXcd& vfield) {
	size_t n = mesh.getNumVerts();
	Eigen::MatrixX3d extrin(n, 3);

	for (size_t i = 0; i < n; i++) {
		auto n = mesh.normal(mesh.verts[i]);
		auto e = mesh.edge2vec(mesh.verts[i]->edge->pair).normalized();
		extrin.row(i) = std::abs(vfield[i]) * gp::util::quatRot(e, n, std::arg(vfield[i]));
	}

	return extrin;
}
}