#pragma once
#include <utility>
#include <queue>

#include "geoproc.hpp"
#include "halfedge.hpp"

#if __has_include("Eigen/Eigen")
#include "Eigen/Core"
#include "Eigen/Sparse"
#define EIGEN_INCLUDE
#endif

namespace gp {
	std::vector<size_t> src2trgPath(const std::vector<std::pair<size_t, double>>& path, size_t src, size_t trg) {
		std::vector<size_t> p;
		size_t t = trg;
		while (t != src) {
			p.push_back(t);
			t = path[t].first;
		}
		p.push_back(src);
		return p;
	}

	std::vector<std::pair<size_t, double>> dijkstraPath(const heds::Mesh& mesh, const std::vector<size_t> src) {
		std::vector<double> min_dist(mesh.getNumVertices(), std::numeric_limits<double>::max());
		for (auto i : src) min_dist[i] = 0;

		auto compare = [](std::pair<size_t, double>& l, std::pair<size_t, double>& r) {return l.second > r.second; };
		std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double>>, decltype(compare)> search;
		for (auto i : src) search.push(std::pair(i, 0.0));
		std::vector<std::pair<size_t, double>> path(mesh.getNumVertices(), { 0, -1.0 });

		while (true) {
			if (search.size() == 0) break;
			auto [v, d] = search.top();
			search.pop();

			for (auto& u : heds::oneringNeighbs(mesh.verts[v])) {
				double dist = (u->pos.glm - mesh.verts[v]->pos.glm).length();
				if (min_dist[u->id] > d + dist) {
					min_dist[u->id] = d + dist;
					search.push(std::pair(u->id, min_dist[u->id]));
					path[u->id] = { v, min_dist[u->id] };
				}
			}
		}

		return path;
	}

	std::vector<std::pair<size_t, double>> p2pDijkstraPath(const heds::Mesh& mesh, const size_t src, const size_t trg) {
		std::vector<double> min_dist(mesh.getNumVertices(), std::numeric_limits<double>::max());
		min_dist[src] = 0;

		using node = std::pair<size_t, double>;
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		search.push(std::pair(src, 0.0));
		std::vector<std::pair<size_t, double>> path(mesh.getNumVertices(), { 0, -1.0 });

		while (true) {
			if (search.size() == 0) break;
			auto [v, d] = search.top();
			if (v == trg) break;
			search.pop();

			for (auto& u : heds::oneringNeighbs(mesh.verts[v])) {
				double dist = d + (u->pos.glm - mesh.verts[v]->pos.glm).length();

				if (min_dist[u->id] > dist) {
					min_dist[u->id] = dist;
					search.push(std::pair(u->id, dist));
					path[u->id] = { v, dist };
				}
			}
		}

		return path;
	}
	
	std::vector<std::pair<size_t, double>> p2pAstarPath(const heds::Mesh& mesh, const size_t src, const size_t trg) {
		std::vector<double> min_dist(mesh.getNumVertices(), std::numeric_limits<double>::max());
		min_dist[src] = (mesh.verts[trg]->pos.glm - mesh.verts[src]->pos.glm).length();

		using node = std::tuple<size_t, double, double>;
		auto compare = [](node& l, node& r) {return std::get<2>(l) > std::get<2>(r); };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		search.push(node(src, 0.0, min_dist[src]));
		std::vector<std::pair<size_t, double>> path(mesh.getNumVertices(), { 0, -1.0 });

		while (true) {
			if (search.size() == 0) break;
			auto [v, d, h] = search.top();
			if (v == trg) return path;
			search.pop();
			
			for (auto& u : heds::oneringNeighbs(mesh.verts[v])) {
				double dist = d + (u->pos.glm - mesh.verts[v]->pos.glm).length();
				double heur = dist + (mesh.verts[trg]->pos.glm - u->pos.glm).length();

				if (min_dist[u->id] > heur) {
					min_dist[u->id] = heur;
					search.push(node(u->id, dist, heur));
					path[u->id] = { v, dist };
				}
			}
		}

		return path;
	}

	#ifdef EIGEN_INCLUDE
	Eigen::VectorXd dijkstraDist(const heds::Mesh& mesh, const std::vector<size_t> src) {
		auto path = dijkstraPath(mesh, src);
		Eigen::VectorXd dist(mesh.getNumVertices());
		for (size_t i = 0; i < dist.size(); i++) dist(i) = path[i].second;
		return dist;
	}

	Eigen::VectorXd geodesicHeatDist(const heds::Mesh& mesh, Eigen::SparseMatrix<double>& lbo, const std::vector<size_t>& src, const double time) {
		size_t nv = mesh.getNumVertices(), nf = mesh.getNumFaces();

		Eigen::SparseMatrix<double> A(nv, nv);
		A.setIdentity();
		A = gp::areaVectorPerVert(mesh).asDiagonal() * A;

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto& i : src) delta(i) = -1.0;

		Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
		solver.compute(A + time * lbo);

		Eigen::VectorXd u = solver.solve(delta);

		Eigen::MatrixXd grad_u = Eigen::MatrixXd::Zero(3, nf);
		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			Eigen::Vector3d normal = mesh.faces[i]->normal<Eigen::Vector3d>();

			auto v0 = mesh.faces[i]->edge->vert;
			auto v1 = mesh.faces[i]->edge->next->vert;
			auto v2 = mesh.faces[i]->edge->next->next->vert;

			auto e0 = v2->pos.eig - v1->pos.eig;
			auto e1 = v0->pos.eig - v2->pos.eig;
			auto e2 = v1->pos.eig - v0->pos.eig;

			grad_u.col(i) += u(v0->id) * (normal.cross(e0));
			grad_u.col(i) += u(v1->id) * (normal.cross(e1));
			grad_u.col(i) += u(v2->id) * (normal.cross(e2));
			grad_u.col(i) /= (2.0 * mesh.faces[i]->area());
		}

		Eigen::MatrixXd x = -grad_u.normalized();
		Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(nv);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto v0 = mesh.faces[i]->edge->vert;
			auto v1 = mesh.faces[i]->edge->next->vert;
			auto v2 = mesh.faces[i]->edge->next->next->vert;

			auto e0 = v2->pos.eig - v1->pos.eig;
			auto e1 = v0->pos.eig - v2->pos.eig;
			auto e2 = v1->pos.eig - v0->pos.eig;

			grad_x(v0->id) += cot(mesh.edges.at({ v2, v0 })) * ((-e1).dot(x.col(i))) +
				cot(mesh.edges.at({ v0, v1 })) * (e2.dot(x.col(i)));
			grad_x(v1->id) += cot(mesh.edges.at({ v0, v1 })) * ((-e2).dot(x.col(i))) +
				cot(mesh.edges.at({ v1, v2 })) * (e0.dot(x.col(i)));
			grad_x(v2->id) += cot(mesh.edges.at({ v1, v2 })) * ((-e0).dot(x.col(i))) +
				cot(mesh.edges.at({ v2, v0 })) * (e1.dot(x.col(i)));
		}
		grad_x *= 0.5;
		grad_x.normalize();

		solver.compute(lbo);
		Eigen::VectorXd dist = solver.solve(grad_x);

		return dist;
	}
	#endif
}