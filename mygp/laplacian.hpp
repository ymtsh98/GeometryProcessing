#pragma once
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <mygp/geoutil.hpp>
#include <mygp/eigutil.hpp>
#include <mygp/halfedge_mesh.hpp>

namespace gp::grad {
/// @brief  
/// @param mesh 
/// @param f : real value function on mesh (vertex-wise)
template <typename T>
Eigen::MatrixX3d surfaceGradient(const heds::Mesh<T>& mesh, const Eigen::VectorXd& f) {
	size_t nf = mesh.getNumFaces();
	Eigen::MatrixX3d grad = Eigen::MatrixX3d::Zero(nf, 3);
	
	for (size_t i = 0; i < nf; i++) {
		auto normal = mesh.normal(mesh.faces[i]);
		auto area2 = normal.norm(); // 0.5 * ||u x v||

		auto& e0 = mesh.faces[i]->edge;
		auto& e1 = e0->next;
		auto& e2 = e1->next;

		auto v0 = e0->vert->id;
		auto v1 = e1->vert->id;
		auto v2 = e2->vert->id;
	
		auto p0 = mesh.position(v0);
		auto p1 = mesh.position(v1);
		auto p2 = mesh.position(v2);
	
		grad.row(i) += f(v0) * (normal.cross(p2 - p1));
		grad.row(i) += f(v1) * (normal.cross(p0 - p2));
		grad.row(i) += f(v2) * (normal.cross(p1 - p0));

		grad.row(i) /= area2 * area2; // normal is non-unit (normal.cross needs unit normal)
	}

	return grad;
}

///	
/// Eigen::MatrixXd triX(tri.rows(), tri.cols()), grad(tri.rows(), tri.cols());
/// triX.col(0) = x(tri.col(0));
/// triX.col(1) = x(tri.col(1));
/// triX.col(2) = x(tri.col(2));
/// auto gm = gp::grad::surfaceGradientMat(srcData.mesh);
/// for (int i = 0; i < tri.rows(); i++) {
/// 	grad.row(i) = triX.row(i) * gm.block<3, 3>(i*3, 0);
/// }
/// 
/// @return  precompute 3|F|x3 matrix for gradient operator
template <typename T>
Eigen::MatrixXd surfaceGradientOp(const heds::Mesh<T>& mesh) {
	size_t nf = mesh.getNumFaces();
	Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(3 * nf, 3);

	for (size_t i = 0; i < nf; i+=3) {
		auto normal = mesh.normal(mesh.faces[i/3]);
		auto area2 = normal.norm(); // area = 0.5 * |n|
		
		auto& e0 = mesh.faces[i/3]->edge;
		auto& e1 = e0->next;
		auto& e2 = e1->next;

		auto v0 = e0->vert->id;
		auto v1 = e1->vert->id;
		auto v2 = e2->vert->id;

		auto p0 = mesh.position(v0);
		auto p1 = mesh.position(v1);
		auto p2 = mesh.position(v2);

		Eigen::Matrix3d m;
		m.col(0) = (normal.cross(p2 - p1)) / area2;
		m.col(1) = (normal.cross(p0 - p2)) / area2;
		m.col(2) = (normal.cross(p1 - p0)) / area2;

		grad.block<3, 3>(i, 0) = m.transpose() * m / area2;
	}

	return grad;
}

// TODO
template <typename T>
Eigen::MatrixXd rot90inTanSpace(const heds::Mesh<T>& mesh, const Eigen::MatrixXd& f) {
	size_t nf = mesh.getNumFaces();
	Eigen::MatrixXd rgrad(nf, 3);
	
	for (size_t i = 0; i < nf; i++) {
		Eigen::Vector3d g = f.row(i);
		auto normal = mesh.normal(mesh.faces[i]).normalized();
		rgrad.row(i) = - g.cross(normal);
	}

	return rgrad;
}

// TODO
template <typename T>
Eigen::SparseMatrix<double> gradientOperator(const heds::Mesh<T>& mesh, const Eigen::VectorXd& areas, const Eigen::MatrixXd& grad) {
	size_t nv = mesh.getNumVerts();
	size_t nf = mesh.getNumFaces();

	Eigen::SparseMatrix<double> mat(nv, nv);
	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(9 * nf);

	for (size_t i = 0; i < nf; i++) {
		Eigen::Vector3d gf = grad.row(i);
		auto normal = mesh.normal(mesh.faces[i]).normalized();
		
		auto& e0 = mesh.faces[i]->edge;
		auto& e1 = e0->next;
		auto& e2 = e1->next;
		
		auto v0 = e0->vert->id;
		auto v1 = e1->vert->id;
		auto v2 = e2->vert->id;
		
		auto p0 = mesh.position(v0);
		auto p1 = mesh.position(v1);
		auto p2 = mesh.position(v2);
		
		double w0 = (normal.cross(p2 - p1)).dot(gf) / 6.0;
		double w1 = (normal.cross(p0 - p2)).dot(gf) / 6.0;
		double w2 = (normal.cross(p1 - p0)).dot(gf) / 6.0;

		triplets.emplace_back(v0, v0, (w0 + w1));
		triplets.emplace_back(v1, v1, (w1 + w2));
		triplets.emplace_back(v2, v2, (w2 + w0));

		triplets.emplace_back(v0, v1, - w1);
		triplets.emplace_back(v1, v0, - w1);
		triplets.emplace_back(v1, v2, - w2);
		triplets.emplace_back(v2, v1, - w2);
		triplets.emplace_back(v2, v0, - w0);
		triplets.emplace_back(v0, v2, - w0);
		
	}

	mat.setFromTriplets(triplets.begin(), triplets.end());

	return areas.cwiseInverse().asDiagonal() * mat;
}

inline Eigen::VectorXd surfaceDivergence(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& field) {
	size_t n = mesh.getNumVerts();
	Eigen::VectorXd div = Eigen::VectorXd::Zero(n);

	if (n == (size_t)field.rows()) {
		for (size_t i = 0; i < n; i++) {
			double l = 0.0;
			Eigen::Vector3d p = mesh.position(i);
			Eigen::Vector3d r = Eigen::Vector3d::Zero();

			auto&& onering = mesh.verts[i]->oneringVindex();
			for (auto u : onering) {
				Eigen::Vector3d e = mesh.position(u) - p;
				r += e;
				l += e.squaredNorm();
			}

			for (auto u : onering) {
				Eigen::Vector3d e = mesh.position(u) - p;
				div(i) += e.dot(field.row(u)) / l;
			}

			div(i) -= r.dot(field.row(i)) / l;
		}
	} else if (mesh.getNumFaces() == (size_t)field.rows()) {
		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto v0 = mesh.faces[i]->edge->vert->id;
			auto v1 = mesh.faces[i]->edge->next->vert->id;
			auto v2 = mesh.faces[i]->edge->next->next->vert->id;

			auto p0 = mesh.position(v0);
			auto p1 = mesh.position(v1);
			auto p2 = mesh.position(v2);

			Eigen::Vector3d f = field.row(i);

			double cd02 = mesh.cot(mesh.faces[i]->edge) *             (p0 - p2).dot(f);
			double cd10 = mesh.cot(mesh.faces[i]->edge->next) *       (p1 - p0).dot(f);
			double cd21 = mesh.cot(mesh.faces[i]->edge->next->next) * (p2 - p1).dot(f);

			div(v0) += 0.5 * (cd10 - cd02);
			div(v1) += 0.5 * (cd21 - cd10);
			div(v2) += 0.5 * (cd02 - cd21);
		}
	}

	return div;
}
}

namespace gp::lbo {
	// cotangent weight laplacian matrix
	// @param mesh	: halfedge mesh
	inline Eigen::SparseMatrix<double> cotanLaplacian(const heds::Mesh<Eigen::MatrixX3d>& mesh) {
		Eigen::SparseMatrix<double> mat(mesh.getNumVerts(), mesh.getNumVerts());
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(9 * mesh.getNumFaces());

		for (const auto& face : mesh.faces) {
			auto& e0 = face->edge;
			auto& e1 = e0->next;
			auto& e2 = e1->next;

			double w0 = 0.5 * mesh.cot(e0);
			double w1 = 0.5 * mesh.cot(e1);
			double w2 = 0.5 * mesh.cot(e2);

			auto v0 = e0->vert->id;
			auto v1 = e1->vert->id;
			auto v2 = e2->vert->id;

			triplets.emplace_back(v0, v0, (w0 + w1));
			triplets.emplace_back(v1, v1, (w1 + w2));
			triplets.emplace_back(v2, v2, (w2 + w0));

			triplets.emplace_back(v0, v1, -w1);
			triplets.emplace_back(v1, v0, -w1);
			triplets.emplace_back(v1, v2, -w2);
			triplets.emplace_back(v2, v1, -w2);
			triplets.emplace_back(v2, v0, -w0);
			triplets.emplace_back(v0, v2, -w0);
		}

		mat.setFromTriplets(triplets.begin(), triplets.end());
		return mat;
	}

	// cotangent weight laplacian matrix (dirichlet boundary condition)
	// @param mesh	: halfedge mesh
	inline Eigen::SparseMatrix<double> cotanLaplacianDirichletBoundary(const heds::Mesh<Eigen::MatrixX3d>& mesh) {
		Eigen::SparseMatrix<double> mat(mesh.getNumVerts(), mesh.getNumVerts());
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(9 * mesh.getNumFaces());
		std::vector<bool> boundary(mesh.getNumVerts());
		for (const auto& vert : mesh.verts) boundary[vert->id] = vert->isBoundary();

		for (const auto& face : mesh.faces) {
			auto& e0 = face->edge;
			auto& e1 = face->edge->next;
			auto& e2 = face->edge->next->next;

			double w0 = 0.5 * mesh.cot(e0);
			double w1 = 0.5 * mesh.cot(e1);
			double w2 = 0.5 * mesh.cot(e2);

			auto v0 = e0->vert->id;
			auto v1 = e1->vert->id;
			auto v2 = e2->vert->id;

			if (!boundary[v0]) {
				triplets.emplace_back(v0, v0, (w0 + w1));
				triplets.emplace_back(v0, v1, -w1);
				triplets.emplace_back(v0, v2, -w0);
			} else { triplets.emplace_back(v0, v0, 1.0); }

			if (!boundary[v1]) {
				triplets.emplace_back(v1, v1, (w1 + w2));
				triplets.emplace_back(v1, v0, -w1);
				triplets.emplace_back(v1, v2, -w2);
			} else { triplets.emplace_back(v1, v1, 1.0); }

			if (!boundary[v2]) {
				triplets.emplace_back(v2, v2, (w2 + w0));
				triplets.emplace_back(v2, v1, -w2);
				triplets.emplace_back(v2, v0, -w0);
			} else { triplets.emplace_back(v2, v2, 1.0); }

		}

		mat.setFromTriplets(triplets.begin(), triplets.end());
		return mat;
	}

	// barycentric areas
	// @param mesh	: halfedge mesh
	inline Eigen::VectorXd barycentricAreas(const heds::Mesh<Eigen::MatrixX3d>& mesh) {
		Eigen::VectorXd areas = Eigen::VectorXd::Zero(mesh.getNumVerts());

		for (const auto& face : mesh.faces) {
			auto area = mesh.area(face);
			for (const auto& v : face->adjacentVindex()) {
				areas[v] += area;
			}
		}

		return areas.array() / 3.0;
	}

	// voronoi cell areas
	// @param mesh	: halfedge mesh
	inline Eigen::VectorXd voronoiAreas(const heds::Mesh<Eigen::MatrixX3d>& mesh) {
		Eigen::VectorXd areas = Eigen::VectorXd::Zero(mesh.getNumVerts());

		for (const auto& face : mesh.faces) {
			double area = mesh.area(face);

			auto& e0 = face->edge;
			auto& e1 = e0->next;
			auto& e2 = e1->next;

			double a = mesh.length(e0), b = mesh.length(e1), c = mesh.length(e2);
			double a2 = a * a, b2 = b * b, c2 = c * c;
			bool isObtuse = gp::util::isObtuseTriangle(a, b, c);
			double cot0 = mesh.cot(e0), cot1 = mesh.cot(e1), cot2 = mesh.cot(e2);

			areas[e0->vert->id] += isObtuse ? (mesh.dot(e0->pair, e1) < 0.0 ? area * 0.5 : area * 0.25) :
				(a2 * cot0 + b2 * cot1) * 0.125;
			areas[e1->vert->id] += isObtuse ? (mesh.dot(e1->pair, e2) < 0.0 ? area * 0.5 : area * 0.25) :
				(b2 * cot1 + c2 * cot2) * 0.125;
			areas[e2->vert->id] += isObtuse ? (mesh.dot(e2->pair, e0) < 0.0 ? area * 0.5 : area * 0.25) :
				(c2 * cot2 + a2 * cot0) * 0.125;
		}

		return areas;
	}

	// return one-ring angles relative to reference edge (mesh.verts[i]->edge) (counter-clockwise)
	inline Eigen::SparseMatrix<double> oneringAngles(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
		Eigen::SparseMatrix<double> phi(mesh.getNumVerts(), mesh.getNumVerts());
		std::vector<Eigen::Triplet<double>> ptriplets;
		ptriplets.reserve(6 * mesh.getNumVerts());

		for (size_t i = 0; i < mesh.verts.size(); i++) {
			auto oring = mesh.verts[i]->oneringVindex();
			// align reference edge
			std::reverse(oring.begin()+1, oring.end());

			Eigen::VectorXd p = Eigen::VectorXd::Zero(oring.size());
			auto v = mesh.position(i);

			for (size_t j = 0, k = oring.size(); j < oring.size(); j++) {
				auto u = mesh.position(oring[j]);
				auto w = mesh.position(oring[(j+1)%k]);
				p[j] = gp::util::angle(u - v, w - v);
			}
			p.array() *= 2.0 * std::numbers::pi / p.sum();

			double theta = 0.0;
			for (size_t j = 0; j < oring.size(); j++) {
				ptriplets.emplace_back((int)i, (int)oring[j], theta);
				theta += p[j];
			}
		}
		phi.setFromTriplets(ptriplets.begin(), ptriplets.end());
		return phi;
	}

	// return one-ring edge filed relative to reference edge (mesh.verts[i]->edge) (counter-clockwise) (aligned neighbs)
	inline Eigen::SparseMatrix<Eigen::dcomplex> oneringAnglesCmplx(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
		auto angles = oneringAngles(mesh);
		Eigen::SparseMatrix<Eigen::dcomplex> phi(mesh.getNumVerts(), mesh.getNumVerts());
		std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;
		triplets.reserve(mesh.edges.size() * 2);

		for (const auto& edge : mesh.edges) {
			auto [u, v] = edge->endpointIndex();
			Eigen::dcomplex ruv = std::polar(1.0, angles.coeff(u, v)), rvu = std::polar(1.0, angles.coeff(v, u));
			rvu = -rvu / ruv; ruv = std::conj(rvu);
			triplets.emplace_back(u, v, ruv);
			triplets.emplace_back(v, u, rvu);
		}
		phi.setFromTriplets(triplets.begin(), triplets.end());

		return phi;
	}

	inline Eigen::SparseMatrix<Eigen::dcomplex> connectionLaplacian(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
		Eigen::SparseMatrix<Eigen::dcomplex> phi = oneringAnglesCmplx(mesh);

		Eigen::SparseMatrix<Eigen::dcomplex> mat(mesh.getNumVerts(), mesh.getNumVerts());
		std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;
		triplets.reserve(9 * mesh.getNumFaces());
		
		for (const auto& face : mesh.faces) {
			auto& e0 = face->edge;
			auto& e1 = e0->next;
			auto& e2 = e1->next;
		
			double w0 = 0.5 * mesh.cot(e0);
			double w1 = 0.5 * mesh.cot(e1);
			double w2 = 0.5 * mesh.cot(e2);

			auto v0 = e0->vert->id;
			auto v1 = e1->vert->id;
			auto v2 = e2->vert->id;
		
			Eigen::dcomplex r01 = phi.coeff(v0, v1), r10 = phi.coeff(v1, v0);
			Eigen::dcomplex r12 = phi.coeff(v1, v2), r21 = phi.coeff(v2, v1);
			Eigen::dcomplex r02 = phi.coeff(v0, v2), r20 = phi.coeff(v2, v0);
		
			triplets.emplace_back(v0, v0, (w0 + w1));
			triplets.emplace_back(v1, v1, (w1 + w2));
			triplets.emplace_back(v2, v2, (w2 + w0));
		
			triplets.emplace_back(v0, v1, -w1 * r01);
			triplets.emplace_back(v1, v0, -w1 * r10);
			triplets.emplace_back(v1, v2, -w2 * r12);
			triplets.emplace_back(v2, v1, -w2 * r21);
			triplets.emplace_back(v2, v0, -w0 * r20);
			triplets.emplace_back(v0, v2, -w0 * r02);
		}

		mat.setFromTriplets(triplets.begin(), triplets.end());
		return mat;
	}
}

namespace gp::grad {

inline Eigen::VectorXcd vertexGradient(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXd& f) {
	size_t nv = mesh.getNumVerts();

	Eigen::VectorXcd grad = Eigen::VectorXcd::Zero(nv);

	auto E = gp::lbo::oneringAngles(mesh);

	for (size_t i = 0; i < nv; i++) {
		Eigen::dcomplex e(0.0, 0.0);
		double sum = 0.0;

		for (const auto& v : mesh.verts[i]->oneringVindex()) {
			double l = (mesh.position(v) - mesh.position(i)).norm();
			double g = f(v) - f(i);
			e += std::polar(g * l, E.coeff(i, v));
			sum += l * l;
		}

		grad(i) = e / sum;
	}

	return grad;
}

inline Eigen::SparseMatrix<Eigen::dcomplex> vertexGradientOp(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
	size_t nv = mesh.getNumVerts();
	auto E = gp::lbo::oneringAngles(mesh);

	Eigen::SparseMatrix<Eigen::dcomplex> grad(nv, nv);
	std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;

	for (size_t i = 0; i < nv; i++) {
		Eigen::dcomplex e(0.0, 0.0);
		double denom = 0.0;

		for (const auto& r : mesh.verts[i]->incidentEdges()) {
			double l = mesh.length(r);
			e += std::polar(l, E.coeff(i, r->pair->vert->id));
			denom += l * l;
		}

		for (const auto& r : mesh.verts[i]->incidentEdges()) {
			double l = mesh.length(r);
			triplets.emplace_back((int)i, (int)r->pair->vert->id, std::polar(l/ denom, E.coeff(i, r->pair->vert->id)));
		}

		triplets.emplace_back((int)i, (int)i, -e / denom);
	}

	grad.setFromTriplets(triplets.begin(), triplets.end());
	return grad;
}

inline Eigen::SparseMatrix<Eigen::dcomplex> vertexDivergenceOp(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
	size_t nv = mesh.getNumVerts();
	auto E = gp::lbo::oneringAngles(mesh);

	Eigen::SparseMatrix<Eigen::dcomplex> div(nv, nv);
	std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;

	for (size_t i = 0; i < nv; i++) {
		Eigen::dcomplex e(0.0, 0.0);
		double denom = 0.0;

		for (const auto& r : mesh.verts[i]->incidentEdges()) {
			double l = mesh.length(r);
			e += std::polar(l, E.coeff(i, r->pair->vert->id));
			denom += l * l;
		}

		for (const auto& r : mesh.verts[i]->incidentEdges()) {
			double l = mesh.length(r);
			triplets.emplace_back((int)i, (int)r->pair->vert->id,
				-std::polar(l/ denom, E.coeff(r->pair->vert->id, i)));
		}

		triplets.emplace_back((int)i, (int)i, e / denom);
	}

	div.setFromTriplets(triplets.begin(), triplets.end());
	return div.conjugate();
}

inline Eigen::SparseMatrix<Eigen::dcomplex> vertexGradientOp(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXd& f) {
	size_t nv = mesh.getNumVerts();
	auto E = gp::lbo::oneringAngles(mesh);
	
	Eigen::SparseMatrix<Eigen::dcomplex> grad(nv, nv);
	std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;
	
	for (size_t i = 0; i < nv; i++) {
		Eigen::dcomplex e(0.0, 0.0);
		double denom = 0.0;
	
		for (const auto& v : mesh.verts[i]->oneringVindex()) {
			double l = (mesh.position(i) - mesh.position(v)).norm();
			e += std::polar(l, E.coeff(i, v));
			denom += l * l;
		}
	
		for (const auto& v : mesh.verts[i]->oneringVindex()) {
			double l = (mesh.position(i) - mesh.position(v)).norm();
			triplets.emplace_back((int)i, (int)v, std::polar(f(v) * l / denom, E.coeff(i, v)));
		}
	
		triplets.emplace_back((int)i, (int)i, -f(i) * e / denom);
	}
	
	grad.setFromTriplets(triplets.begin(), triplets.end());
	return grad;
}

}

namespace gp::lbo {
/// 
/// Each element has an angle between the reference edge and incident edge.
/// col(0 - max incidnet size), index-0 corresponds to minimal angle difference edge.
/// 
/// for (size_t i = 0; i < data.n; i++) {
///   Eigen::Vector3d n = data.vn.row(i);
///   Eigen::Vector3d p = data.v.row(i);
///   Eigen::Vector3d e = data.mesh.edge2vec(data.mesh.verts[i]->edge->pair).normalized();
///   
///   for (int j = 0; j < refEdges.cols(); j++) {
///     double r = std::abs(refEdges(i, j));
///     if (r < 1e-20) break;
///     Eigen::Vector3d d = gp::util::quatRot(e, n, std::arg(refEdges(i, j))) * r;
///   }
/// }
/// 
inline Eigen::MatrixXcd oneringAnglesFromRefs(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXcd& refEdges) {
	size_t n = mesh.getNumVerts();

	Eigen::MatrixXcd angles = Eigen::MatrixXcd::Zero(n, 3);

	for (size_t i = 0; i < n; i++) {
		auto e = mesh.edge2vec(mesh.verts[i]->edge);
		auto n = mesh.normal(mesh.verts[i]);

		Eigen::Vector3d extrin = gp::util::quatRot(e, n, std::arg(refEdges[i]));
		auto iedges = mesh.verts[i]->incidentEdges();
		std::reverse(iedges.begin()+1, iedges.end());
		if (iedges.size() >(size_t)angles.cols()) angles = gp::eig::expandMatWithZero(angles, angles.rows(), iedges.size());

		double closestAngle = 10.0;
		size_t closestIndex = 0;
		for (size_t j = 0; j < iedges.size(); j++) {
			auto edge = mesh.edge2vec(iedges[j]);
			auto angle = gp::util::angle(extrin, edge);
			if (closestAngle > angle) {
				closestAngle = angle;
				closestIndex = j;
			}
		}

		double oneringAngle = 0.0;
		for (size_t j = 0; j < iedges.size(); j++) {
			size_t k = (j + closestIndex) % iedges.size();
			auto edge = mesh.edge2vec(iedges[k]);
			oneringAngle += gp::util::angle(extrin, edge, n);
			angles(i, j) = std::polar(edge.norm(), oneringAngle);
			extrin = edge;
		}
	}

	return angles;
}
}

namespace gp::lbo {
/// @brief Polygon Laplacian Made Simple. [Bunge A., et al. 2020].
class VirtualVertexLaplacian {
	class VirtualVertexSolver {
		Eigen::MatrixXd A;
		Eigen::VectorXd b;

	public:
		VirtualVertexSolver() {}

		/// @brief to find a point minimizing the area of the	triangle fan.
		/// @param verts : counter-clockwise ordered verteices of a polygon.
		/// @return weight : (opt p = sigma {w_i verts_i}).
		Eigen::VectorXd solve4area(const Eigen::MatrixX3d& verts) {
			size_t n = (int)verts.rows();
			std::vector<Eigen::Vector3d> vdiff(n), vcross(n*n);
			for (size_t i = 0; i < n; i++) vdiff[i] = verts.row((i+1)%n) - verts.row(i);
			for (size_t i = 0; i < n; i++)
				for (size_t j = 0; j < n; j++) vcross[i * n + j] = verts.row(i).cross(vdiff[j]);

			A.resize(n+1, n);
			A.setZero();

			b.resize(n+1);
			b.setZero();

			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < n; j++) {
					for (size_t r = 0; r < n; r++) {
						//A(i, j) += (verts.row(j).cross(vdiff[r])).dot(verts.row(i).cross(vdiff[r]));
						A(i, j) += vcross[j * n + r].dot(vcross[i * n + r]);
					}
				}
			}

			A.array() *= 2.0;
			A.bottomRows(1).array() = 1.0;

			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < n; j++) {
					//b[i] +=	(verts.row(i).cross(vdiff[j])).dot(vdiff[j].cross(verts.row(j)));
					b[i] +=	vcross[i * n + j].dot(-vcross[j * n + j]);
				}
			}

			b.array() *= 2.0;
			b[n] = -1.0;

			return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-b);
		}
	};

	Eigen::SparseMatrix<double> P;
	Eigen::MatrixX3d vrVerts;
	gp::heds::Mesh<Eigen::MatrixX3d> vrMesh;

	void init(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
		size_t nv = mesh.getNumVerts(), nf = mesh.getNumFaces(), nvf = nv + nf, ntri = 0;
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.resize(nf * 5);
		for (size_t i = 0; i < nv; i++) triplets.emplace_back((int)i, (int)i, 1.0);

		auto verts = mesh.copyPosition();
		vrVerts = verts;
		vrVerts.conservativeResize(nvf, 3);
		std::vector<std::array<size_t, 3>> virtualTris;
		virtualTris.reserve(nf * 5);

		VirtualVertexSolver vvs;

		for (size_t i = 0, t = 0; i < nf; i++) {
			auto vidx = mesh.faces[i]->adjacentVindex();
			if (vidx.size() == 3) {
				ntri++;
				virtualTris.push_back(std::array<size_t, 3>{ vidx[0], vidx[1], vidx[2] });
			} else {
				Eigen::MatrixX3d X = verts(vidx, Eigen::all);
				Eigen::VectorXd w = vvs.solve4area(X);

				size_t j = nv + t++;
				vrVerts.row(j) = X.transpose() * w;

				for (size_t r = 0; r < vidx.size(); r++) {
					triplets.emplace_back((int)j, (int)vidx[r], w[r]);
					virtualTris.push_back(std::array<size_t, 3>{ vidx[r], vidx[(r+1)%vidx.size()], j });
				}
			}
		}

		vrVerts.conservativeResize(nvf - ntri, 3);
		P.resize(nvf - ntri, nv);
		P.setFromTriplets(triplets.begin(), triplets.end());

		vrMesh.build(vrVerts, virtualTris);
	}

public:
	VirtualVertexLaplacian(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
		init(mesh);
	}

	Eigen::VectorXd voronoiAreas() const {
		auto vrAreas = gp::lbo::voronoiAreas(vrMesh);
		return P.transpose() * (vrAreas.asDiagonal() * (P * Eigen::VectorXd::Ones(P.cols())));
	}

	Eigen::SparseMatrix<double> laplacian() const {
		auto vrLbo = gp::lbo::cotanLaplacian(vrMesh);
		return P.transpose() * vrLbo * P;
	}
};

}