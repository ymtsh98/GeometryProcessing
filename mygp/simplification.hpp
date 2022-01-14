#pragma once
#include <queue>
#include <utility>
#include <mygp/halfedge_mesh.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace gp::impl::qem {
	template <typename T>
	Eigen::Matrix4d computeQEM(const heds::Mesh<T>& mesh, const size_t i) {
		Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();

		auto p = mesh.getPosition<Eigen::Vector3d>(i);
		for (const auto& f : mesh.verts[i]->adjacentFaces()) {
			auto n = mesh.normal<Eigen::Vector3d>(f);
			auto a = mesh.area(f);
			Eigen::Vector4d q(n[0], n[1], n[2], -n.dot(p));
			Q += a * q * q.transpose();
		}

		return Q;
	}

	template <typename T>
	Eigen::Matrix4d computeQEM(const heds::Mesh<T>& mesh, const Eigen::MatrixXd& normals, const Eigen::VectorXd& areas, const size_t i) {
		Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();

		auto p = mesh.getPosition<Eigen::Vector3d>(i);
		auto e = mesh.verts[i]->edge;

		do {
			Eigen::Vector3d n = normals.row(e->face->id);
			Eigen::Vector4d q(n[0], n[1], n[2], -n.dot(p));
			Q += areas(e->face->id) * q * q.transpose();
			e = e->next->pair;
		} while (e->face != nullptr && mesh.verts[i]->edge != e);

		return Q;
	}

	std::pair<double, Eigen::Vector3d> computeOptPosition(
		const Eigen::Matrix4d& Q1, const Eigen::Matrix4d& Q2){
		//,	const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
		//Eigen::Matrix3d P = 0.0000001 * Eigen::Matrix3d::Identity();
		Eigen::Matrix4d Q = Q1 + Q2;
		Eigen::Matrix3d A = Q.block(0, 0, 3, 3);// + P;
		Eigen::Vector3d b = Q.block(0, 3, 3, 1);// + 0.5 * (p1+p2) * 0.0000001;// + P.block(0, 3, 3, 1);
		//Eigen::FullPivLU<Eigen::Matrix3d> solver(A);
		Eigen::LDLT<Eigen::Matrix3d> solver(A);
		double cost = -1.0;
		Eigen::Vector3d p;
		//if (solver.isInvertible()) {
		if (solver.info() == Eigen::Success) {
			p = solver.solve(-b);
			cost = p.transpose() * A * p + 2.0 * b.dot(p) + Q(3, 3);
		}

		return std::make_pair(cost, p);
	}

	template <typename T>
	bool normalFlip(const heds::Mesh<T>& mesh, const size_t i,
									const Eigen::Vector3d& optv, const double threshold) {

		Eigen::Vector3d p = mesh.getPosition<Eigen::Vector3d>(i);
		auto neighbs = mesh.verts[i]->oneringNeighbs();
		size_t l = neighbs.size();

		for (size_t j = 0; j < (l - mesh.verts[i]->isBoundary()); j++) {
			Eigen::Vector3d e1 = mesh.getPosition<Eigen::Vector3d>(neighbs[j]->id) - p;
			Eigen::Vector3d e2 = mesh.getPosition<Eigen::Vector3d>(neighbs[(j+1)%l]->id) - p;
			Eigen::Vector3d n1 = e2.cross(e1).normalized();

			e1 =  mesh.getPosition<Eigen::Vector3d>(neighbs[j]->id) - optv;
			e2 = mesh.getPosition<Eigen::Vector3d>(neighbs[(j+1)%l]->id) - optv;
			Eigen::Vector3d n2 = e2.cross(e1).normalized();
			if (n1.dot(n2) < threshold) return true;
		}

		return false;
	}

	template <typename T>
	std::pair<double, Eigen::Vector3d> getCostPos(const heds::Mesh<T>& mesh,
																								const std::vector<Eigen::Matrix4d>& qem,
																								const size_t i, const size_t j, const double threshold) {
		constexpr double LARGE_NUMBER = 1e+6;

		std::pair<double, Eigen::Vector3d> cp;

		cp = computeOptPosition(qem[i], qem[j]);
			//, mesh.getPosition<Eigen::Vector3d>(i), mesh.getPosition<Eigen::Vector3d>(j));

		if (cp.first < 0.0) {
			cp.first = LARGE_NUMBER;
			switch ((i^j)%3) {
			case 0:
				cp.second = mesh.getPosition<Eigen::Vector3d>(i);
				break;
			case 1:
				cp.second = mesh.getPosition<Eigen::Vector3d>(j);
				break;
			case 2:
				cp.second = 0.5 * (mesh.getPosition<Eigen::Vector3d>(i) +
													 mesh.getPosition<Eigen::Vector3d>(j));
			}
		}

		if (mesh.verts[i]->isBoundary() || mesh.verts[j]->isBoundary()) {
			cp.second = mesh.verts[i]->isBoundary() ?
				(mesh.verts[j]->isBoundary() ?
				 0.5 * (mesh.getPosition<Eigen::Vector3d>(i) +
								mesh.getPosition<Eigen::Vector3d>(j)) :
				 mesh.getPosition<Eigen::Vector3d>(i))
				: mesh.getPosition<Eigen::Vector3d>(j);
		}

		if (threshold < 1.0) {
			cp.first += normalFlip(mesh, i, cp.second, threshold) * 5.0 * LARGE_NUMBER;
			cp.first += normalFlip(mesh, j, cp.second, threshold) * 5.0 * LARGE_NUMBER;
		}

		return cp;
	}


}

namespace gp {
	// warning : rewrites binded positions
	template <typename T>
	void qemSimplification(heds::Mesh<T>& mesh, const size_t numTarget, const double nflipThres=1.0) {
		size_t nv = mesh.getNumVertices();

		// --------------- initialization ---------------

		std::vector<size_t> vertexCounter(nv, 0);

		std::vector<Eigen::Matrix4d> quadricErrorMetrices(nv);
		#pragma omp parallel for
		for (int i = 0; i < nv; i++) {
			quadricErrorMetrices[i] = impl::qem::computeQEM(mesh, i);
		}

		using node = std::tuple<double, size_t, size_t, heds::pte, Eigen::Vector3d>;
		auto compare = [](auto& l, auto& r) { return std::get<0>(l) > std::get<0>(r); };
		std::priority_queue<node, std::vector<node>, decltype(compare)> queue;

		for (const auto& emap : mesh.edges) {
			auto [pi, pj] = emap.first;
			auto [cost, optv] = impl::qem::getCostPos(mesh, quadricErrorMetrices, pi, pj, nflipThres);
			queue.emplace(cost, 0, 0, emap.second, optv);
		}

		// --------------- main loop ---------------

		while (nv > numTarget && !queue.empty()) {
			auto [cost, c1, c2, edge, optv] = queue.top();
			queue.pop();

			auto pi = edge->vert->id, pj = edge->pair->vert->id;

			if (vertexCounter[pi] != c1 || vertexCounter[pj] != c2 ||
					edge != edge->pair->pair || !edge->isContractible()) continue;

			vertexCounter[pj] = std::numeric_limits<size_t>::max();

			mesh.setPosition(pi, optv);
			heds::ptv v = mesh.verts[pi];
			edge->contract(v);

			vertexCounter[pi]++;
			quadricErrorMetrices[pi] += quadricErrorMetrices[pj];
			auto ine = v->incidentEdges();
			for (const auto& e : ine) vertexCounter[e->pair->vert->id]++;

			for (const auto& e : ine) {
				pi = v->id; pj = e->pair->vert->id;

				std::tie(cost, optv) = impl::qem::getCostPos(mesh, quadricErrorMetrices, pi, pj, nflipThres);
				queue.emplace(cost, vertexCounter[pi], vertexCounter[pj], e, optv);

				if (e->face == nullptr) continue;

				auto en = e->next->next;
				pi = en->vert->id; pj = en->pair->vert->id;
				std::tie(cost, optv) = impl::qem::getCostPos(mesh, quadricErrorMetrices, pi, pj, nflipThres);
				queue.emplace(cost, vertexCounter[pi], vertexCounter[pj], en, optv);
			}

			nv--;
		}

		// --------------- post process ---------------
		std::deque<size_t> deque;
		for (size_t i = 0; i < vertexCounter.size(); i++) {
			if (vertexCounter[i] == std::numeric_limits<size_t>::max()) {
				deque.emplace_back(i);
			} else if (deque.size() != 0) {
				auto j = deque.front();
				deque.pop_front();
				mesh.swapPosition(i, j);
				std::swap(mesh.verts[i], mesh.verts[j]);
				mesh.verts[j]->id = (uint32_t)j;
				deque.emplace_back(i);
			}
		}

		mesh.verts.resize(nv);
		mesh.resizePosition(nv);

		mesh.edges.clear();

		size_t nf = 0;
		deque.clear();
		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			if (mesh.faces[i]->isInvalid()) {
				deque.emplace_back(i);
			} else {
				nf++;

				auto e0 = mesh.faces[i]->edge;
				auto e = mesh.faces[i]->edge;
				do {
					heds::eKey key(e->vert->id, e->pair->vert->id);
					if (!mesh.edges.contains(heds::keyPair(key))) mesh.edges[key] = e;
					e = e->next;
				} while (e0 != e);

				if (deque.size() == 0) continue;
				auto j = deque.front();
				deque.pop_front();
				std::swap(mesh.faces[i], mesh.faces[j]);
				mesh.faces[j]->id = (uint32_t)j;
				deque.emplace_back(i);
			}
		}

		mesh.faces.resize(nf);
	}

	// TODO : sloppy implementation
	template <typename T>
	std::pair<Eigen::MatrixXd, std::vector<uint32_t>> vclustSimplification(heds::Mesh<T>& mesh, const size_t numGrid) {
		Eigen::Vector3d minCorner = mesh.getPosition<Eigen::Vector3d>(0);
		Eigen::Vector3d maxCorner = mesh.getPosition<Eigen::Vector3d>(0);
		for (size_t i = 0; i < mesh.getNumVertices(); i++) {
			auto v = mesh.getPosition<Eigen::Vector3d>(i);
			for (size_t j = 0; j < 3; j++) {
				minCorner[j] = std::min(minCorner[j], v[j]);
				maxCorner[j] = std::max(maxCorner[j], v[j]);
			}
		}
		Eigen::Vector3d cellSize = (maxCorner - minCorner).array() / (double)numGrid;
		for (size_t i = 0; i < 3; i++) if (abs(cellSize[i]) < 1e-5) cellSize[i] = 1.0;

		std::vector<Eigen::Matrix4d> quadricErrorMetrices(mesh.getNumVertices());
		#pragma omp parallel for
		for (int i = 0; i < quadricErrorMetrices.size(); i++) {
			quadricErrorMetrices[i] = impl::qem::computeQEM(mesh, i);
		}

		Eigen::MatrixXd representPos = Eigen::MatrixXd::Zero(numGrid * numGrid * numGrid, 3);
		std::vector<Eigen::Matrix4d> representQEM(numGrid * numGrid * numGrid, Eigen::Matrix4d::Zero());
		std::vector<size_t> count(numGrid * numGrid * numGrid, 0);

		auto gridIndex = [&numGrid, &minCorner, &cellSize](Eigen::Vector3d v) {
			v.array() -= minCorner.array();
			v.array() /= cellSize.array();
			v = Eigen::floor(v.array()).min((double)(numGrid - 1));
			return numGrid * numGrid * (size_t)v[0] + numGrid * (size_t)v[1] + (size_t)v[2];
		};

		std::vector<uint32_t> indices;
		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto verts = mesh.faces[i]->adjacentVerts();
			auto v1 = mesh.getPosition<Eigen::Vector3d>(verts[0]->id);
			auto v2 = mesh.getPosition<Eigen::Vector3d>(verts[1]->id);
			auto v3 = mesh.getPosition<Eigen::Vector3d>(verts[2]->id);
			auto i1 = gridIndex(v1);
			auto i2 = gridIndex(v2);
			auto i3 = gridIndex(v3);

			representQEM[i1] += quadricErrorMetrices[verts[0]->id];
			representQEM[i2] += quadricErrorMetrices[verts[1]->id];
			representQEM[i3] += quadricErrorMetrices[verts[2]->id];
			//representPos.row(i1) += v1;
			//representPos.row(i2) += v2;
			//representPos.row(i3) += v3;
			count[i1]++;	count[i2]++;	count[i3]++;

			if (i1 != i2 && i1 != i3 && i2 != i3) {
				indices.emplace_back((uint32_t)i1);
				indices.emplace_back((uint32_t)i2);
				indices.emplace_back((uint32_t)i3);
			}
		}

		Eigen::LDLT<Eigen::Matrix3d> solver;
		for (size_t i = 0; i < representPos.rows(); i++) {
			if (count[i] == 0) continue;
			//representPos.row(i) /= (double)count[i];

			Eigen::Matrix3d A = representQEM[i].block(0, 0, 3, 3);
			Eigen::Vector3d b = representQEM[i].block(0, 3, 3, 1);
			solver.compute(A);
			representPos.row(i) = solver.solve(-b);
		}

		return std::make_pair(representPos, indices);
	}

}