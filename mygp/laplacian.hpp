#pragma once
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>

namespace gp::lbo {
	// cotangent weight laplacian matrix
	// @param mesh	: halfedge mesh
	template<typename T>
	Eigen::SparseMatrix<double> cotanLaplacian(const heds::Mesh<T>& mesh) {
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
	template<typename T>
	Eigen::SparseMatrix<double> cotanLaplacianDirichletBoundary(const heds::Mesh<T>& mesh) {
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
	template<typename T>
	Eigen::VectorXd barycentricAreas(const heds::Mesh<T>& mesh) {
		Eigen::VectorXd areas = Eigen::VectorXd::Zero(mesh.getNumVerts());

		for (const auto& face : mesh.faces) {
			auto area = mesh.area(face) / 3.0;
			for (const auto& vert : face->adjacentVerts()) {
				areas[vert->id] += area;
			}
		}

		return areas;
	}

	// voronoi cell areas
	// @param mesh	: halfedge mesh
	template<typename T>
	Eigen::VectorXd voronoiAreas(const heds::Mesh<T>& mesh) {
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


}