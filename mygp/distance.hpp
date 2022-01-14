#pragma once
#include <queue>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/laplacian.hpp>

namespace gp::impl {
	double fastMarchingDistance(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3,
															const double d1, const double d2) {
		auto u = v1 - v3;
		auto v = v2 - v3;
		double dotUV = u.dot(v);

		double d = d2 - d1, a = v.norm(), b = u.norm();
		if (d2 < d1) {
			d = -d;
			std::swap(a, b);
		}
		dotUV /= (a * b);

		double A = a * a + b * b - 2 * a * b * dotUV;
		double B = 2 * b * d * (a * dotUV - b);
		double C = b * b * (d * d - a * a * (1 - dotUV * dotUV));

		double r = std::max(0.0, B*B - 4 * A * C);
		double t = (-B + sqrt(r)) / (2 * A);

		double CD = b * (t - d) / t;

		double dist;
		if (t > d &&(CD < a/dotUV) && (CD > a * dotUV)) {
			dist = (d1 < d2 ? d1 : d2) + t;
		} else {
			double di = u.norm() + d1;
			double dj = v.norm() + d2;
			dist = di < dj ? di : dj;
		}
		return dist;
	}
}

namespace gp {
	template <typename T1, typename T2=size_t>
	std::vector<std::pair<size_t, double>> dijkstraPath(const heds::Mesh<T1>& mesh, 
																											const std::vector<T2>& src, 
																											const size_t trg = std::numeric_limits<size_t>::max()) {
		using node = std::pair<size_t, double>;
		std::vector<double> minDist(mesh.getNumVertices(), std::numeric_limits<double>::max());
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		std::vector<node> path(mesh.getNumVertices(), { 0, -1.0 });

		for (const auto i : src) {
			minDist[i] = 0;
			search.push(std::make_pair(i, 0.0));
			path[i].first = i;
			path[i].second = 0.0;
		}

		while (true) {
			if (search.size() == 0) break;
			auto [v, d] = search.top();
			if (v == trg) break;
			search.pop();

			for (const auto& u : mesh.verts[v]->oneringNeighbs()) {
				double dist = (mesh.getPosition<Eigen::Vector3d>(u->id) - mesh.getPosition<Eigen::Vector3d>(v)).norm();
				if (minDist[u->id] > d + dist) {
					minDist[u->id] = d + dist;
					search.push(std::make_pair(u->id, minDist[u->id]));
					path[u->id] ={ v, minDist[u->id] };
				}
			}
		}

		return path;
	}

	template<typename T1, typename T2=size_t>
	Eigen::VectorXd dijkstraDist(const heds::Mesh<T1>& mesh, const std::vector<T2>& src, 
															 const size_t trg = std::numeric_limits<size_t>::max()) {
		auto path = dijkstraPath(mesh, src, trg);
		Eigen::VectorXd dist(mesh.getNumVertices());
		for (Eigen::Index i = 0; i < dist.size(); i++) dist(i) = path[i].second;
		return dist;
	}

	template <typename T1, typename T2=size_t>
	std::vector<std::pair<size_t, double>> aStarPath(const heds::Mesh<T1>& mesh,
																											const std::vector<T2>& src,
																											const size_t trg) {
		using node = std::tuple<size_t, double, double>;
		std::vector<double> minHeur(mesh.getNumVertices(), std::numeric_limits<double>::max());
		auto compare = [](node& l, node& r) {return std::get<2>(l) > std::get<2>(r); };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		std::vector<std::pair<size_t, double>> path(mesh.getNumVertices(), { 0, -1.0 });

		for (const auto i : src) {
			minHeur[i] = (mesh.getPosition<Eigen::Vector3d>(trg) - mesh.getPosition<Eigen::Vector3d>(i)).norm();
			search.push(std::make_tuple(i, 0.0, minHeur[i]));
			path[i].first = i;
			path[i].second = 0.0;
		}

		while (true) {
			if (search.size() == 0) break;
			auto [v, d, h] = search.top();
			if (v == trg) break;
			search.pop();

			for (const auto& u : mesh.verts[v]->oneringNeighbs()) {
				double dist = d + (mesh.getPosition<Eigen::Vector3d>(u->id) - mesh.getPosition<Eigen::Vector3d>(v)).norm();
				double heur = dist + (mesh.getPosition<Eigen::Vector3d>(u->id) - mesh.getPosition<Eigen::Vector3d>(trg)).norm();
				if (minHeur[u->id] > heur) {
					minHeur[u->id] = heur;
					search.push(std::make_tuple(u->id, dist, minHeur[u->id]));
					path[u->id] ={ v, dist };
				}
			}
		}

		return path;
	}

	std::vector<size_t> src2trgPath(const std::vector<std::pair<size_t, double>>& path, const size_t src, const size_t trg) {
		std::vector<size_t> p;
		size_t t = trg;
		while (t != src) {
			p.emplace_back(t);
			t = path[t].first;
		}
		p.emplace_back(src);
		return p;
	}

	template<typename T1, typename T2=size_t>
	Eigen::VectorXd aStarDist(const heds::Mesh<T1>& mesh, const std::vector<T2>& src,
															 const size_t trg) {
		auto path = aStarPath(mesh, src, trg);
		Eigen::VectorXd dist(mesh.getNumVertices());
		for (Eigen::Index i = 0; i < dist.size(); i++) dist(i) = path[i].second;
		return dist;
	}

	template <typename T1, typename T2=size_t>
	Eigen::VectorXd fastMarchingDist(const heds::Mesh<T1>& mesh,
																	 const std::vector<T2>& src,
																	 const size_t trg = std::numeric_limits<size_t>::max(),
																	 const double limit = std::numeric_limits<double>::max()) {
		using node = std::pair<size_t, double>;

		size_t nv = mesh.getNumVertices();
		std::vector<std::pair<double, bool>> minDist(nv, std::make_pair(std::numeric_limits<double>::max(), false));

		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;

		for (const auto i : src) {
			minDist[i] = std::make_pair(0.0, true);
			Eigen::Vector3d s = mesh.getPosition<Eigen::Vector3d>(i);
			for (const auto& v : mesh.verts[i]->oneringNeighbs()) {
				double d = (mesh.getPosition<Eigen::Vector3d>(v->id) - s).norm();
				minDist[v->id].first = d;
				search.emplace(v->id, d);
			}
		}

		while (!search.empty()) {
			auto [v, d] = search.top();
			if (v == trg || d > limit) break;
			search.pop();
			if (minDist[v].second) continue;
			minDist[v].second = true;

			for (const auto& f : mesh.verts[v]->adjacentFaces()) {
				auto e0 = f->edge;
				while (e0->vert->id != v) e0 = e0->next;
				auto u = e0->next->vert->id;
				auto k = e0->next->next->vert->id;

				if (minDist[u].second == minDist[k].second) continue;
				if (minDist[k].second) std::swap(u, k);

				double dist = impl::fastMarchingDistance(
					mesh.getPosition<Eigen::Vector3d>(v),
					mesh.getPosition<Eigen::Vector3d>(u),
					mesh.getPosition<Eigen::Vector3d>(k),
					minDist[v].first, minDist[u].first);

				if (dist < minDist[k].first) {
					minDist[k].first = dist;
					search.emplace(k, dist);
				}
			}
		}

		Eigen::VectorXd dist(nv);
		for (size_t i = 0; i < nv; i++) {
			if (minDist[i].first == std::numeric_limits<double>::max()) dist(i) = -1.0;
			else dist(i) = minDist[i].first;
		}
		return dist;
	}

	template <typename T1, typename T2=size_t>
	Eigen::VectorXd aStarFastMarchingDist(const heds::Mesh<T1>& mesh,
																	 const std::vector<T2>& src,
																	 const size_t trg) {
		using node = std::pair<size_t, double>;
		size_t nv = mesh.getNumVertices();
		std::vector<std::pair<double, bool>> minDist(nv, 
			std::make_pair(std::numeric_limits<double>::max(), false));
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;

		Eigen::Vector3d vtrg = mesh.getPosition<Eigen::Vector3d>(trg);
		for (const auto i : src) {
			minDist[i] = std::make_pair(0.0, true);
			Eigen::Vector3d s = mesh.getPosition<Eigen::Vector3d>(i);
			for (const auto& v : mesh.verts[i]->oneringNeighbs()) {
				Eigen::Vector3d p = mesh.getPosition<Eigen::Vector3d>(v->id);
				double d = (p - s).norm();
				double h = d + (vtrg - p).norm();
				minDist[v->id].first = d;
				search.emplace(v->id, h);
			}
		}

		while (!search.empty()) {
			auto [v, h] = search.top();
			if (v == trg) break;
			search.pop();
			if (minDist[v].second) continue;
			minDist[v].second = true;

			for (const auto& f : mesh.verts[v]->adjacentFaces()) {
				auto e0 = f->edge;
				while (e0->vert->id != v) e0 = e0->next;
				auto u = e0->next->vert->id;
				auto k = e0->next->next->vert->id;

				if (minDist[u].second == minDist[k].second) continue;
				if (minDist[k].second) std::swap(u, k);

				auto p = mesh.getPosition<Eigen::Vector3d>(k);
				double dist = impl::fastMarchingDistance(
					mesh.getPosition<Eigen::Vector3d>(v), mesh.getPosition<Eigen::Vector3d>(u),
					p, minDist[v].first, minDist[u].first);

				double heur = dist + (p - vtrg).norm();
				if (dist < minDist[k].first) {
					minDist[k].first = dist;
					search.emplace(k, heur);
				}
			}
		}

		Eigen::VectorXd dist(nv);
		for (size_t i = 0; i < nv; i++) {
			if (minDist[i].first == std::numeric_limits<double>::max()) dist(i) = -1.0;
			else dist(i) = minDist[i].first;
		}
		return dist;
	}

	// geodesic heat distance
	template <typename T1, typename T2=size_t>
	Eigen::VectorXd geodesicHeatDist(const heds::Mesh<T1>& mesh, 
																	 const Eigen::SparseMatrix<double>& lbo, 
																	 const Eigen::VectorXd& areas,
																	 const std::vector<T2>& src, 
																	 const double time, 
																	 const bool dirichletCondition = false) {
		size_t nv = mesh.getNumVertices(), nf = mesh.getNumFaces();

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto i : src) delta(i) = 1.0;

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		Eigen::SparseMatrix<double> A(nv, nv);
		A.setIdentity();
		A.diagonal() *= areas.diagonal();

		solver.compute(A - time * lbo);

		Eigen::VectorXd u = solver.solve(delta);

		if (dirichletCondition) {
			solver.compute(A - time * gp::cotanLaplacianDirichletBoundary(mesh));
			u += solver.solve(delta);
			u.array() *= 0.5;
		}

		Eigen::MatrixXd gu = Eigen::MatrixXd::Zero(3, nf);
		Eigen::MatrixXd x(3, nf);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto normal = mesh.normal<Eigen::Vector3d>(mesh.faces[i]);

			auto& v0 = mesh.faces[i]->edge->vert;
			auto& v1 = mesh.faces[i]->edge->next->vert;
			auto& v2 = mesh.faces[i]->edge->next->next->vert;

			auto p0 = mesh.getPosition<Eigen::Vector3d>(v0->id);
			auto p1 = mesh.getPosition<Eigen::Vector3d>(v1->id);
			auto p2 = mesh.getPosition<Eigen::Vector3d>(v2->id);

			gu.col(i) += u(v0->id) * (normal.cross(p2 - p1));
			gu.col(i) += u(v1->id) * (normal.cross(p0 - p2));
			gu.col(i) += u(v2->id) * (normal.cross(p1 - p0));
			gu.col(i) /= normal.norm();
			
			x.col(i) = -gu.col(i).normalized();
			//x.col(i) = -gu.col(i).stableNormalized();
		}

		Eigen::VectorXd gx = Eigen::VectorXd::Zero(nv);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto& v0 = mesh.faces[i]->edge->vert;
			auto& v1 = mesh.faces[i]->edge->next->vert;
			auto& v2 = mesh.faces[i]->edge->next->next->vert;

			auto p0 = mesh.getPosition<Eigen::Vector3d>(v0->id);
			auto p1 = mesh.getPosition<Eigen::Vector3d>(v1->id);
			auto p2 = mesh.getPosition<Eigen::Vector3d>(v2->id);

			double cd02 = mesh.cot(mesh.faces[i]->edge) * (p0 - p2).dot(x.col(i));
			double cd10 = mesh.cot(mesh.faces[i]->edge->next) * (p1 - p0).dot(x.col(i));
			double cd21 = mesh.cot(mesh.faces[i]->edge->next->next) * (p2 - p1).dot(x.col(i));

			gx(v0->id) += cd10 - cd02;
			gx(v1->id) += cd21 - cd10;
			gx(v2->id) += cd02 - cd21;
		}

		gx *= 0.5;
		gx.normalize();

		solver.compute(lbo);
		return solver.solve(gx);
	}


	template <typename T1, typename T2>
	Eigen::VectorXd geodesicHeatDist(const heds::Mesh<T1>& mesh,
		const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver1, 
		const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver2,
		const std::vector<T2>& src) {

		size_t nv = mesh.getNumVertices(), nf = mesh.getNumFaces();

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto i : src) delta(i) = 1.0;

		Eigen::VectorXd u = solver1.solve(delta);

		Eigen::MatrixXd gu = Eigen::MatrixXd::Zero(3, nf);
		Eigen::MatrixXd x(3, nf);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto normal = mesh.normal<Eigen::Vector3d>(mesh.faces[i]);

			auto& v0 = mesh.faces[i]->edge->vert;
			auto& v1 = mesh.faces[i]->edge->next->vert;
			auto& v2 = mesh.faces[i]->edge->next->next->vert;

			auto p0 = mesh.getPosition<Eigen::Vector3d>(v0->id);
			auto p1 = mesh.getPosition<Eigen::Vector3d>(v1->id);
			auto p2 = mesh.getPosition<Eigen::Vector3d>(v2->id);

			gu.col(i) += u(v0->id) * (normal.cross(p2 - p1));
			gu.col(i) += u(v1->id) * (normal.cross(p0 - p2));
			gu.col(i) += u(v2->id) * (normal.cross(p1 - p0));
			gu.col(i) /= normal.norm();

			x.col(i) = -gu.col(i).normalized();
		}

		Eigen::VectorXd gx = Eigen::VectorXd::Zero(nv);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto& v0 = mesh.faces[i]->edge->vert;
			auto& v1 = mesh.faces[i]->edge->next->vert;
			auto& v2 = mesh.faces[i]->edge->next->next->vert;

			auto p0 = mesh.getPosition<Eigen::Vector3d>(v0->id);
			auto p1 = mesh.getPosition<Eigen::Vector3d>(v1->id);
			auto p2 = mesh.getPosition<Eigen::Vector3d>(v2->id);

			double cd02 = mesh.cot(mesh.faces[i]->edge) * (p0 - p2).dot(x.col(i));
			double cd10 = mesh.cot(mesh.faces[i]->edge->next) * (p1 - p0).dot(x.col(i));
			double cd21 = mesh.cot(mesh.faces[i]->edge->next->next) * (p2 - p1).dot(x.col(i));

			gx(v0->id) += cd10 - cd02;
			gx(v1->id) += cd21 - cd10;
			gx(v2->id) += cd02 - cd21;
		}

		gx *= 0.5;
		gx.normalize();

		return solver2.solve(gx);
	}

	// geodesic heat distance
	template <typename T1, typename T2=size_t>
	Eigen::MatrixXd geodesicGradient(const heds::Mesh<T1>& mesh,
																	 const Eigen::SparseMatrix<double>& lbo,
																	 const Eigen::VectorXd& areas,
																	 const std::vector<T2>& src,
																	 const double time,
																	 const bool dirichletCondition = false) {
		size_t nv = mesh.getNumVertices(), nf = mesh.getNumFaces();

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto i : src) delta(i) = 1.0;

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		Eigen::SparseMatrix<double> A(nv, nv);
		A.setIdentity();
		A.diagonal() *= areas.diagonal();

		solver.compute(A - time * lbo);

		Eigen::VectorXd u = solver.solve(delta);

		if (dirichletCondition) {
			solver.compute(A - time * gp::cotanLaplacianDirichletBoundary(mesh));
			u += solver.solve(delta);
			u.array() *= 0.5;
		}

		Eigen::MatrixXd gu = Eigen::MatrixXd::Zero(3, nf);
		Eigen::MatrixXd x(3, nf);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto normal = mesh.normal<Eigen::Vector3d>(mesh.faces[i]);

			auto& v0 = mesh.faces[i]->edge->vert;
			auto& v1 = mesh.faces[i]->edge->next->vert;
			auto& v2 = mesh.faces[i]->edge->next->next->vert;

			auto p0 = mesh.getPosition<Eigen::Vector3d>(v0->id);
			auto p1 = mesh.getPosition<Eigen::Vector3d>(v1->id);
			auto p2 = mesh.getPosition<Eigen::Vector3d>(v2->id);

			gu.col(i) += u(v0->id) * (normal.cross(p2 - p1));
			gu.col(i) += u(v1->id) * (normal.cross(p0 - p2));
			gu.col(i) += u(v2->id) * (normal.cross(p1 - p0));
			gu.col(i) /= normal.norm();

			x.col(i) = -gu.col(i).normalized();
		}

		return gu.transpose();
	}

}