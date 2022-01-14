#pragma once
#if __has_include("Eigen/Eigen")
#define EIGEN_INCLUDE
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Sparse"
#endif

#include "halfedge.hpp"
#include "union_find.hpp"

namespace gp {
	template <typename T>
	double cot(const T edge) {
		auto e1 = edge->vert->pos.eig - edge->next->vert->pos.eig;
		auto e2 = edge->prev->vert->pos.eig - edge->next->vert->pos.eig;

		double dot = e1.dot(e2);
		double cross_norm = (e1.cross(e2)).norm();

		return dot / cross_norm;
	}

	template <typename T1, typename T2>
	auto connectedMesh(const T1& pos, const T2& faces) {
		UnionFind uf(pos.size());
		for (auto& face : faces) {
			for (size_t i = 1; i < face.size(); i++) uf.unite(face[0], face[i]);
		}

		std::unordered_map<size_t, std::vector<size_t>> s;
		for (size_t i = 0; i < pos.size(); i++) {
			s[uf.root(i)].push_back(i);
		}

		return s;
	}

	Eigen::SparseMatrix<double> areaSparseMat(const heds::Mesh& mesh, double div = 3.0) {
		using Triplet = Eigen::Triplet<double>;
		Eigen::SparseMatrix<double> mat(mesh.getNumVertices(), mesh.getNumVertices());
		std::vector<Triplet> triplets;

		Eigen::VectorXd count = Eigen::VectorXd::Zero(mesh.getNumVertices());
		for (auto& f : mesh.faces) {
			auto a = f->area();
			auto v0 = f->edge->vert;
			auto v1 = f->edge->next->vert;
			auto v2 = f->edge->next->next->vert;

			triplets.push_back(Triplet(v0->id, v0->id, a));
			triplets.push_back(Triplet(v1->id, v1->id, a));
			triplets.push_back(Triplet(v2->id, v2->id, a));

			count(v0->id) += 1.0;
			count(v1->id) += 1.0;
			count(v2->id) += 1.0;
		}

		mat.setFromTriplets(triplets.begin(), triplets.end());
		if (div == 0.0) mat.diagonal().array() /= count.array();
		else mat.diagonal().array() /= div;

		return mat;
	}

	Eigen::VectorXd areaVectorPerFace(const heds::Mesh& mesh, float div = 3.0) {
		Eigen::VectorXd areas(mesh.getNumFaces());
		for (auto& face : mesh.faces) areas[face->id] = face->area() / div;
		return areas;
	}

	Eigen::VectorXd areaVectorPerVert(const heds::Mesh& mesh, float div = 3.0) {
		Eigen::VectorXd areas = Eigen::VectorXd::Zero(mesh.getNumVertices());
		for (auto& face : mesh.faces) {
			auto a = face->area();
			auto v0 = face->edge->vert;
			auto v1 = face->edge->next->vert;
			auto v2 = face->edge->next->next->vert;
			areas[v0->id] += a;
			areas[v1->id] += a;
			areas[v2->id] += a;
		}
		areas.array() /= div;
		return areas;
	}

	auto getDataFromSparseMat(const Eigen::SparseMatrix<double>& mat) {
		std::vector<size_t> rows, cols;
		std::vector<double> vals;
		for (int k = 0; k < mat.outerSize(); k++) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
				rows.push_back(it.row());
				cols.push_back(it.col());
				vals.push_back(it.value());
			}
		}

		return std::tuple(std::move(rows), std::move(cols), std::move(vals));
	}

	Eigen::VectorXd heatDiffusionSmoothing(const Eigen::SparseMatrix<double>& lbo, Eigen::VectorXd heat, const double time) {
		Eigen::SparseMatrix<double> eye(lbo.rows(), lbo.rows());
		eye.setIdentity();
		Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
		solver.compute(eye + time * lbo);

		Eigen::VectorXd smoothed_heat = solver.solve(heat);
		return smoothed_heat;
	}
}