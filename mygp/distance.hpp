#pragma once
#include <queue>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/laplacian.hpp>

namespace gp::dist::impl {
	inline double fastMarchingDistance(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3,
															const double d1, const double d2) {
		auto u = v1 - v3;
		auto v = v2 - v3;
		double dotUV = u.dot(v);

		double d = d2 - d1, a = v.norm(), b = u.norm(), alpha = a, beta = b;
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
			double di = beta + d1;
			double dj = alpha + d2;
			dist = di < dj ? di : dj;
		}
		return dist;
	}
}

namespace gp::dist {
	template <typename T=size_t>
	std::vector<std::pair<size_t, double>> dijkstraPath(const heds::Mesh<Eigen::MatrixX3d>& mesh, 
		const std::vector<T>& src, 
		const size_t trg = std::numeric_limits<size_t>::max(), 
		const double limit = std::numeric_limits<double>::max()) {
		using node = std::pair<size_t, double>;
		std::vector<double> minDist(mesh.getNumVerts(), std::numeric_limits<double>::max());
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		std::vector<node> path(mesh.getNumVerts(), { 0, -1.0 });
		std::vector<bool> used(minDist.size(), false);

		for (const auto i : src) {
			if (minDist[i] == 0) continue;
			minDist[i] = 0;
			search.push(std::make_pair(i, 0.0));
			path[i].first = i;
			path[i].second = 0.0;
			if (i == trg) return path;
		}

		while (!search.empty()) {
			auto [v, d] = search.top();
			if (v == trg || d > limit) break;
			search.pop();
			
			if (used[v]) continue;

			used[v] = true;

			for (const auto& u : mesh.verts[v]->oneringVindex()) {
				if (used[u]) continue;
				double dist = (mesh.position(u) - mesh.position(v)).norm();
				if (minDist[u] > d + dist) {
					minDist[u] = d + dist;
					search.push(std::make_pair(u, minDist[u]));
					path[u] ={ v, minDist[u] };
				}
			}
		}

		return path;
	}

	template<typename T=size_t>
	Eigen::VectorXd dijkstraDist(const heds::Mesh<Eigen::MatrixX3d>& mesh, const std::vector<T>& src, 
		const size_t trg = std::numeric_limits<size_t>::max(), 
		const double limit = std::numeric_limits<double>::max()) {
		auto&& path = dijkstraPath(mesh, src, trg, limit);
		Eigen::VectorXd dist(mesh.getNumVerts());
		for (Eigen::Index i = 0; i < dist.size(); i++) dist(i) = path[i].second;
		return dist;
	}

	template <typename T=size_t>
	std::vector<std::pair<size_t, double>> aStarPath(const heds::Mesh<Eigen::MatrixX3d>& mesh,
																											const std::vector<T>& src,
																											const size_t trg) {
		using node = std::tuple<size_t, double, double>;
		std::vector<double> minHeur(mesh.getNumVerts(), std::numeric_limits<double>::max());
		auto compare = [](node& l, node& r) {return std::get<2>(l) > std::get<2>(r); };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		std::vector<std::pair<size_t, double>> path(mesh.getNumVerts(), { 0, -1.0 });
		std::vector<bool> used(minHeur.size(), false);

		for (const auto i : src) {
			if (path[i].second == 0.0) continue;
			minHeur[i] = (mesh.position(trg) - mesh.position(i)).norm();
			search.push(std::make_tuple(i, 0.0, minHeur[i]));
			path[i].first = i;
			path[i].second = 0.0;
			if (i == trg) return path;
		}

		while (!search.empty()) {
			auto [v, d, h] = search.top();
			if (v == trg) break;
			search.pop();

			if (used[v]) continue;
			used[v] = true;

			for (const auto& u : mesh.verts[v]->oneringVindex()) {
				if (used[u]) continue;
				double dist = d + (mesh.position(u) - mesh.position(v)).norm();
				double heur = dist + (mesh.position(u) - mesh.position(trg)).norm();
				if (minHeur[u] > heur) {
					minHeur[u] = heur;
					search.push(std::make_tuple(u, dist, minHeur[u]));
					path[u] ={ v, dist };
				}
			}
		}

		return path;
	}

	template<typename T=size_t>
	Eigen::VectorXd aStarDist(const heds::Mesh<Eigen::MatrixX3d>& mesh, const std::vector<T>& src,
															 const size_t trg) {
		auto&& path = aStarPath(mesh, src, trg);
		Eigen::VectorXd dist(mesh.getNumVerts());
		for (Eigen::Index i = 0; i < dist.size(); i++) dist(i) = path[i].second;
		return dist;
	}

	template<typename T=size_t>
	double aStarDistValue(const heds::Mesh<Eigen::MatrixX3d>& mesh, const std::vector<T>& src, const size_t trg) {
		return aStarPath(mesh, src, trg)[trg].second;
	}

	inline 	std::vector<size_t> src2trgPath(const std::vector<std::pair<size_t, double>>& path, const size_t src, const size_t trg) {
		std::vector<size_t> p;
		size_t t = trg;
		while (t != src) {
			p.emplace_back(t);
			t = path[t].first;
		}
		p.emplace_back(src);
		return p;
	}

	template <typename T=size_t>
	Eigen::VectorXd fastMarchingDist(const heds::Mesh<Eigen::MatrixX3d>& mesh, const std::vector<T>& src,
		const size_t trg = std::numeric_limits<size_t>::max(),
		const double limit = std::numeric_limits<double>::max()) {

		using node = std::pair<size_t, double>;

		size_t nv = mesh.getNumVerts();
		std::vector<double> minDist(nv, std::numeric_limits<double>::max());
		std::vector<bool> used(nv, false);
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;
		bool trgInSrc = false;

		for (const auto i : src) {
			if (minDist[i] == 0.0) continue;

			minDist[i] = 0.0;
			used[i] = true;
			trgInSrc |= (i == trg);

			Eigen::Vector3d s = mesh.position(i);
			for (const auto& v : mesh.verts[i]->oneringVindex()) {
				double d = (mesh.position(v) - s).norm();
				if (minDist[v] > d) {
					minDist[v] = d;
					search.emplace(v, d);
				}
			}
		}

		if (trgInSrc) search = decltype(search)();

		while (!search.empty()) {
			auto [v, d] = search.top();
			if (v == trg || d > limit) break;
			search.pop();

			if (used[v]) continue;
			used[v] = true;

			for (const auto& e0 : mesh.verts[v]->incidentEdges()) {
				if (e0->face == nullptr) continue;
				auto u = e0->next->vert->id;
				auto k = e0->next->next->vert->id;

				if (used[u] == used[k]) continue;
				if (used[k]) std::swap(u, k);

				double dist = impl::fastMarchingDistance(
					mesh.position(v),
					mesh.position(u),
					mesh.position(k),
					minDist[v], minDist[u]);

				if (dist < minDist[k]) {
					minDist[k] = dist;
					search.emplace(k, dist);
				}
			}
		}

		Eigen::VectorXd dist(nv);
		for (size_t i = 0; i < nv; i++) {
			if (minDist[i] == std::numeric_limits<double>::max()) dist(i) = -1.0;
			else dist(i) = minDist[i];
		}
		return dist;
	}

	template <typename T=size_t>
	Eigen::VectorXd aStarFastMarchingDist(const heds::Mesh<Eigen::MatrixX3d>& mesh,
																	 const std::vector<T>& src,
																	 const size_t trg) {
		using node = std::pair<size_t, double>;
		size_t nv = mesh.getNumVerts();
		std::vector<double> minDist(nv, std::numeric_limits<double>::max());
		std::vector<bool> used(nv, false);
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;

		Eigen::Vector3d vtrg = mesh.position(trg);
		for (const auto i : src) {
			if (minDist[i] == 0.0) continue;

			minDist[i] = 0.0;
			used[i] = true;

			Eigen::Vector3d s = mesh.position(i);
			for (const auto& v : mesh.verts[i]->oneringVindex()) {
				Eigen::Vector3d p = mesh.position(v);
				double d = (p - s).norm();
				if (minDist[v] > d) {
					double h = d + (vtrg - p).norm();
					minDist[v] = d;
					search.emplace(v, h);
				}
			}
		}

		while (!search.empty()) {
			auto [v, h] = search.top();
			if (used[trg]) break;
			search.pop();
			if (used[v]) continue;
			used[v] = true;

			for (const auto& e0 : mesh.verts[v]->incidentEdges()) {
				if (e0->face == nullptr) continue;
				auto u = e0->next->vert->id;
				auto k = e0->next->next->vert->id;

				if (used[u] == used[k]) continue;
				if (used[k]) std::swap(u, k);

				auto p = mesh.position(k);
				double dist = impl::fastMarchingDistance(
					mesh.position(v), mesh.position(u),
					p, minDist[v], minDist[u]);

				double heur = dist + (p - vtrg).norm();
				if (dist < minDist[k]) {
					minDist[k] = dist;
					search.emplace(k, heur);
				}
			}
		}

		Eigen::VectorXd dist(nv);
		for (size_t i = 0; i < nv; i++) {
			if (minDist[i] == std::numeric_limits<double>::max()) dist(i) = -1.0;
			else dist(i) = minDist[i];
		}
		return dist;
	}

	template <typename T=size_t>
	double aStarFastMarchingDistValue(const heds::Mesh<Eigen::MatrixX3d>& mesh,
		const std::vector<T>& src,
		const size_t trg) {
		using node = std::pair<size_t, double>;
		size_t nv = mesh.getNumVerts();
		std::vector<double> minDist(nv, std::numeric_limits<double>::max());
		std::vector<bool> used(nv, false);
		auto compare = [](node& l, node& r) {return l.second > r.second; };
		std::priority_queue<node, std::vector<node>, decltype(compare)> search;

		Eigen::Vector3d vtrg = mesh.position(trg);
		for (const auto i : src) {
			if (minDist[i] == 0.0) continue;

			minDist[i] = 0.0;
			used[i] = true;

			Eigen::Vector3d s = mesh.position(i);
			for (const auto& v : mesh.verts[i]->oneringVindex()) {
				Eigen::Vector3d p = mesh.position(v);
				double d = (p - s).norm();
				if (minDist[v] > d) {
					double h = d + (vtrg - p).norm();
					minDist[v] = d;
					search.emplace(v, h);
				}
			}
		}

		while (!search.empty()) {
			auto [v, h] = search.top();
			if (used[trg]) break;
			search.pop();
			if (used[v]) continue;
			used[v] = true;

			for (const auto& e0 : mesh.verts[v]->incidentEdges()) {
				if (e0->face == nullptr) continue;
				auto u = e0->next->vert->id;
				auto k = e0->next->next->vert->id;

				if (used[u] == used[k]) continue;
				if (used[k]) std::swap(u, k);

				auto p = mesh.position(k);
				double dist = impl::fastMarchingDistance(
					mesh.position(v), mesh.position(u),
					p, minDist[v], minDist[u]);

				double heur = dist + (p - vtrg).norm();
				if (dist < minDist[k]) {
					minDist[k] = dist;
					search.emplace(k, heur);
				}
			}
		}

		return minDist[trg];
	}

	template <typename T = size_t>
	Eigen::VectorXd virtualSourcePropDist(const heds::Mesh<Eigen::MatrixX3d>& mesh, const std::vector<T>& src) {
		constexpr double THRESHOLD_c = 5.1424;
		constexpr double THRESHOLD_g = 4.20638;
		constexpr double THRESHOLD_h = 0.504201;
		constexpr double THRESHOLD_hg = 2.84918;
		constexpr double LAMBDA[16] ={
			0.320991, 0.446887, 0.595879,  0.270094,
			0.236679, 0.159685,  0.0872932, 0.434132,
			1.0, 0.726262, 0.0635997, 0.0515979,
			0.56903, 0.0447586, 0.0612103, 0.718198 };
		
		size_t nv = mesh.getNumVerts();
		size_t nf = mesh.getNumFaces();
		size_t ne = mesh.getNumEdges();
		
		double iteration = 0.0;

		double ael = 0.0;
		std::vector<double> face2centerDist(nf, std::numeric_limits<double>::max());
		std::vector<double> face2extraDist(nf, 0.);
		std::vector<double> edgeLength(ne);
		std::vector<double> edge2vertexDist(2*ne, std::numeric_limits<double>::max());
		std::vector<bool> behindMask(2*ne, false);

		#pragma omp parallel for reduction(+:ael)
		for (int i = 0; i < mesh.getNumEdges(); i++) {
			double length = mesh.length(mesh.edges[i]);
			edgeLength[mesh.edges[i]->id / 2] = length;
			ael += length;
		}
		ael /= (double)mesh.edges.size();
		
		std::vector<heds::pte> searchFrom, searchTo;

		std::vector<bool> used(nv, false);
		for (const auto s : src) {
			if (used[s]) continue;
			for (const auto& e2 : mesh.verts[s]->incidentEdges()) {
				if (e2->face == nullptr) continue;

				auto& e0 = e2->next;
				auto& e1 = e0->next;

				auto p0 = mesh.position(e0->vert->id);
				auto p1 = mesh.position(e1->vert->id);
				auto p2 = mesh.position(e2->vert->id);

				Eigen::Vector3d centroid = (p0 + p1 + p2) / 3.0;

				auto h0 = e0->id;
				auto h1 = e1->id;
				auto h2 = e2->id;
				auto d0 = edgeLength[h0/2];
				auto d1 = edgeLength[h1/2];
				auto d2 = edgeLength[h2/2];

				edge2vertexDist[h2] = std::min(edge2vertexDist[h2], 0.0);
				edge2vertexDist[h0] = std::min(edge2vertexDist[h0], util::pow2(d0));
				edge2vertexDist[h1] = std::min(edge2vertexDist[h1], util::pow2(d2));

				auto f = e0->face->id;
				face2extraDist[f] = 0.0;
				face2centerDist[f] = std::min(face2centerDist[f], (centroid - p2).norm());
				searchFrom.emplace_back(e1);
			}
			used[s] = true;
		}
		
		while (!searchFrom.empty()) {
			iteration += ael;
			searchTo.clear();

			for (size_t i = 0; i < searchFrom.size(); i++) {
				auto& srcEdge = searchFrom[i];

				auto& trgEdge = srcEdge->pair;

				if (trgEdge->face == nullptr) continue;

				auto& trgEdgeNext = trgEdge->next;
				auto& trgEdgePrev = trgEdgeNext->next;
				
				auto e1 = edgeLength[trgEdge->id/2];
				auto e2 = edgeLength[trgEdgePrev->id/2];
				auto e3 = edgeLength[trgEdgeNext->id/2];

				double& refDist = face2centerDist[trgEdge->face->id];

				bool isBehind = behindMask[srcEdge->id];
				auto d1 = edge2vertexDist[srcEdge->id];
				auto d2 = edge2vertexDist[srcEdge->next->next->id]; // (prev)
				auto sigmaT = face2extraDist[srcEdge->face->id];

				double dist;
				double dA, dB, dC;

				double px = (e1 * e1 + (e2 * e2 - e3 * e3)) / 2.0 / e1;
				double py = util::zeroBoundSqrt(e2 * e2 - px * px);

				double sx = (e1 * e1 + (d1 - d2)) / 2.0 / e1;
				double sy = util::zeroBoundSqrt(d1 - sx * sx);

				double ds1 = sqrt(sx * sx + sy * sy);
				double ds2 = sqrt(util::pow2(sx - e1) + sy * sy);

				double cx = (px + e1) / 3.0;
				double cy = py / 3.0;

				double dc1 = sqrt(cx * cx + cy * cy);
				double dc2 = sqrt(util::pow2(cx - e1) + cy * cy);

				if (isBehind) {
					if ((dc1 + ds1) < (dc2 + ds2)) {
						dA = 0.0;
						dB = e1 * e1;
						dC = e2 * e2;

						sigmaT += ds1;
						dist = sigmaT + dc1;
					} else {
						dA = e1 * e1;
						dB = 0.0;
						dC = e3 * e3;

						sigmaT += ds2;
						dist = sigmaT + dc2;
					}
				} else {
					bool bendLeft = false, bendRight = false;

					double maxe = std::max(e1, std::max(e2, e3));
					double mine = std::min(e1, std::min(e2, e3));
					double tc = maxe, tg = maxe;
					double th = py, thg = py;
					bool b0 = tc >   (THRESHOLD_c * e1);
					bool b1 = tg >   (THRESHOLD_g * mine);
					bool b2 = th   < (THRESHOLD_h* e1);
					bool b3 = thg  < (THRESHOLD_hg* maxe);
					double l = LAMBDA[b0 + b1 * 2 + b2 * 4 + b3 * 8];
					double qx = px * (1.0 - l) + cx * l;
					double qy = py * (1.0 - l) + cy * l;

					double ttx = qx * sy + sx * qy;

					bendLeft = ttx < 0;
					bendRight = ttx > e1 * (qy + sy);

					if (bendLeft) {
						dA = 0.0;
						dB = e1 * e1;
						dC = e2 * e2;

						sigmaT += ds1;
						dist = sigmaT + dc1;
					} else if (bendRight) {
						dA = e1 * e1;
						dB = 0.0;
						dC = e3 * e3;

						sigmaT += ds2;
						dist = sigmaT + dc2;
					} else {
						dA = d1;
						dB = d2;
						dC = util::pow2(px - sx) + util::pow2(py + sy);

						dist = sigmaT + sqrt(util::pow2(cx - sx) + util::pow2(cy + sy));

						if (py * sx + px * sy < 0) behindMask[trgEdgePrev->id] = true;
						if (py * (e1 - sx) - (px - e1) * sy < 0) behindMask[trgEdgeNext->id] = true;
					}
				}

				if (dist < refDist) {
					refDist = dist;

					face2extraDist[trgEdge->face->id] = sigmaT;
					edge2vertexDist[trgEdge->id] = dB;
					edge2vertexDist[trgEdgePrev->id] = dA;
					edge2vertexDist[trgEdgeNext->id] = dC;

					auto& searchNext = dist < iteration ? searchFrom : searchTo;
					searchNext.emplace_back(trgEdgePrev);
					searchNext.emplace_back(trgEdgeNext);
				}
			}

			std::swap(searchFrom, searchTo);
		}

		Eigen::VectorXd geodist(nv);
		#pragma omp parallel for
		for (int i = 0; i < nv; i++) {
			double mindist = std::numeric_limits<double>::max();

			for (const auto& e : mesh.verts[i]->incidentEdges()) {
				if (e->face == nullptr) continue;
				double dist = edge2vertexDist[e->id];

				if (dist != std::numeric_limits<double>::max()) {
					mindist = std::min(mindist, face2extraDist[e->face->id] + sqrt(dist));
				} else {
					mindist = -1.0;
					break;
				}
			}

			geodist(i) = mindist;
		}

		return geodist;
	}
 
	/// @param mesh 
	/// @param srcVertsId : three vertex index per face 
	/// @param barycoords : barycentric coordinates per face
	/// @return 
	template <typename T1 = std::vector<size_t>, typename T2 = std::vector<double>>
	Eigen::VectorXd virtualSourcePropDist(const heds::Mesh<Eigen::MatrixX3d>& mesh, 
		const std::vector<T1>& srcVertsId, const std::vector<T2>& barycoords) {
		constexpr double THRESHOLD_c = 5.1424;
		constexpr double THRESHOLD_g = 4.20638;
		constexpr double THRESHOLD_h = 0.504201;
		constexpr double THRESHOLD_hg = 2.84918;
		constexpr double LAMBDA[16] ={
			0.320991, 0.446887, 0.595879,  0.270094,
			0.236679, 0.159685,  0.0872932, 0.434132,
			1.0, 0.726262, 0.0635997, 0.0515979,
			0.56903, 0.0447586, 0.0612103, 0.718198 };

		size_t nv = mesh.getNumVerts();
		size_t nf = mesh.getNumFaces();
		size_t ne = mesh.getNumEdges();

		double iteration = 0.0;

		double ael = 0.0;
		std::vector<double> face2centerDist(nf, std::numeric_limits<double>::max());
		std::vector<double> face2extraDist(nf, 0.);
		std::vector<double> edgeLength(ne);
		std::vector<double> edge2vertexDist(2*ne, std::numeric_limits<double>::max());
		std::vector<bool> behindMask(2*ne, false);

		#pragma omp parallel for reduction(+:ael)
		for (int i = 0; i < mesh.getNumEdges(); i++) {
			double length = mesh.length(mesh.edges[i]);
			edgeLength[mesh.edges[i]->id / 2] = length;
			ael += length;
		}
		ael /= (double)mesh.edges.size();

		std::vector<heds::pte> searchFrom, searchTo;

		heds::pte e0 = nullptr;
		uint32_t cache1 = 0, cache2 = 0;
		for (size_t i = 0; i < srcVertsId.size(); i++) {
			uint32_t k1 = (uint32_t)srcVertsId[i][2], k2 = (uint32_t)srcVertsId[i][0];
			if (cache1 != k1 || cache2 != k2) {
				cache1 = k1; cache2 = k2;
				e0 = mesh.halfedge(k1, k2);
				if (e0 == nullptr) e0 = mesh.halfedge(k2, k1);
			}
			auto& e1 = e0->next;
			auto& e2 = e1->next;

			auto p0 = mesh.position(e0->vert->id);
			auto p1 = mesh.position(e1->vert->id);
			auto p2 = mesh.position(e2->vert->id);
			
			Eigen::Vector3d vsource =
				(double)barycoords[i][0] * p0 +
				(double)barycoords[i][1] * p1 +
				(double)barycoords[i][2] * p2;

			Eigen::Vector3d centroid = (p0 + p1 + p2) / 3.0;
			
			edge2vertexDist[e0->id] = std::min(edge2vertexDist[e0->id], (p0 - vsource).squaredNorm());
			edge2vertexDist[e1->id] = std::min(edge2vertexDist[e1->id], (p1 - vsource).squaredNorm());
			edge2vertexDist[e2->id] = std::min(edge2vertexDist[e2->id], (p2 - vsource).squaredNorm());
			
			auto f = e0->face->id;
			face2extraDist[f] = 0.0;
			double cpdist = (centroid - vsource).norm();
			if (face2centerDist[f] > cpdist) {
				face2centerDist[f] = cpdist;

				searchFrom.emplace_back(e0);
				searchFrom.emplace_back(e1);
				searchFrom.emplace_back(e2);
			}
		}

		while (!searchFrom.empty()) {
			iteration += ael;
			searchTo.clear();

			for (size_t i = 0; i < searchFrom.size(); i++) {
				auto& srcEdge = searchFrom[i];

				auto& trgEdge = srcEdge->pair;

				if (trgEdge->face == nullptr) continue;

				auto& trgEdgeNext = trgEdge->next;
				auto& trgEdgePrev = trgEdgeNext->next;

				auto e1 = edgeLength[trgEdge->id/2];
				auto e2 = edgeLength[trgEdgePrev->id/2];
				auto e3 = edgeLength[trgEdgeNext->id/2];

				double& refDist = face2centerDist[trgEdge->face->id];

				bool isBehind = behindMask[srcEdge->id];
				auto d1 = edge2vertexDist[srcEdge->id];
				auto d2 = edge2vertexDist[srcEdge->next->next->id]; // (prev)
				auto sigmaT = face2extraDist[srcEdge->face->id];

				double dist;
				double dA, dB, dC;

				double px = (e1 * e1 + (e2 * e2 - e3 * e3)) / 2.0 / e1;
				double py = util::zeroBoundSqrt(e2 * e2 - px * px);

				double sx = (e1 * e1 + (d1 - d2)) / 2.0 / e1;
				double sy = util::zeroBoundSqrt(d1 - sx * sx);

				double ds1 = sqrt(sx * sx + sy * sy);
				double ds2 = sqrt(util::pow2(sx - e1) + sy * sy);

				double cx = (px + e1) / 3.0;
				double cy = py / 3.0;

				double dc1 = sqrt(cx * cx + cy * cy);
				double dc2 = sqrt(util::pow2(cx - e1) + cy * cy);

				if (isBehind) {
					if ((dc1 + ds1) < (dc2 + ds2)) {
						dA = 0.0;
						dB = e1 * e1;
						dC = e2 * e2;

						sigmaT += ds1;
						dist = sigmaT + dc1;
					} else {
						dA = e1 * e1;
						dB = 0.0;
						dC = e3 * e3;

						sigmaT += ds2;
						dist = sigmaT + dc2;
					}
				} else {
					bool bendLeft = false, bendRight = false;

					double maxe = std::max(e1, std::max(e2, e3));
					double mine = std::min(e1, std::min(e2, e3));
					double tc = maxe, tg = maxe;
					double th = py, thg = py;
					bool b0 = tc >   (THRESHOLD_c * e1);
					bool b1 = tg >   (THRESHOLD_g * mine);
					bool b2 = th   < (THRESHOLD_h* e1);
					bool b3 = thg  < (THRESHOLD_hg* maxe);
					double l = LAMBDA[b0 + b1 * 2 + b2 * 4 + b3 * 8];
					double qx = px * (1.0 - l) + cx * l;
					double qy = py * (1.0 - l) + cy * l;

					double ttx = qx * sy + sx * qy;

					bendLeft = ttx < 0;
					bendRight = ttx > e1 * (qy + sy);

					if (bendLeft) {
						dA = 0.0;
						dB = e1 * e1;
						dC = e2 * e2;

						sigmaT += ds1;
						dist = sigmaT + dc1;
					} else if (bendRight) {
						dA = e1 * e1;
						dB = 0.0;
						dC = e3 * e3;

						sigmaT += ds2;
						dist = sigmaT + dc2;
					} else {
						dA = d1;
						dB = d2;
						dC = util::pow2(px - sx) + util::pow2(py + sy);

						dist = sigmaT + sqrt(util::pow2(cx - sx) + util::pow2(cy + sy));

						if (py * sx + px * sy < 0) behindMask[trgEdgePrev->id] = true;
						if (py * (e1 - sx) - (px - e1) * sy < 0) behindMask[trgEdgeNext->id] = true;
					}
				}

				if (dist < refDist) {
					refDist = dist;

					face2extraDist[trgEdge->face->id] = sigmaT;
					edge2vertexDist[trgEdge->id] = dB;
					edge2vertexDist[trgEdgePrev->id] = dA;
					edge2vertexDist[trgEdgeNext->id] = dC;

					auto& searchNext = dist < iteration ? searchFrom : searchTo;
					searchNext.emplace_back(trgEdgePrev);
					searchNext.emplace_back(trgEdgeNext);
				}
			}

			std::swap(searchFrom, searchTo);
		}

		Eigen::VectorXd geodist(nv);
		#pragma omp parallel for
		for (int i = 0; i < nv; i++) {
			double mindist = std::numeric_limits<double>::max();

			for (const auto& e : mesh.verts[i]->incidentEdges()) {
				if (e->face == nullptr) continue;
				double dist = edge2vertexDist[e->id];

				if (dist != std::numeric_limits<double>::max()) {
					mindist = std::min(mindist, face2extraDist[e->face->id] + sqrt(dist));
				} else {
					mindist = -1.0;
					break;
				}
			}

			geodist(i) = mindist;
		}

		return geodist;
	}

	// geodesic heat distance
	template <typename T=size_t>
	Eigen::VectorXd geodesicHeatDist(const heds::Mesh<Eigen::MatrixX3d>& mesh,
		const Eigen::SparseMatrix<double>& lbo, 
		const Eigen::VectorXd& areas,
		const std::vector<T>& src, 
		const double time, 
		const bool dirichletCondition = false) {
		size_t nv = mesh.getNumVerts(), nf = mesh.getNumFaces();

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto i : src) delta(i) = 1.0;

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		Eigen::SparseMatrix<double> A(nv, nv);
		A.setIdentity();
		A.diagonal().array() *= areas.array();

		solver.compute(A + time * lbo);

		Eigen::VectorXd u = solver.solve(delta);

		if (dirichletCondition) {
			solver.compute(A + time * gp::lbo::cotanLaplacianDirichletBoundary(mesh));
			u += solver.solve(delta);
			u.array() *= 0.5;
		}

		Eigen::MatrixXd gu = Eigen::MatrixXd::Zero(3, nf);
		Eigen::MatrixXd x(3, nf);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto normal = mesh.normal(mesh.faces[i]);
			auto area2 = normal.norm(); // 0.5 * ||u x v||

			auto v0 = mesh.faces[i]->edge->vert->id;
			auto v1 = mesh.faces[i]->edge->next->vert->id;
			auto v2 = mesh.faces[i]->edge->next->next->vert->id;

			auto p0 = mesh.position(v0);
			auto p1 = mesh.position(v1);
			auto p2 = mesh.position(v2);

			gu.col(i) += u(v0) * (normal.cross(p2 - p1));
			gu.col(i) += u(v1) * (normal.cross(p0 - p2));
			gu.col(i) += u(v2) * (normal.cross(p1 - p0));
			gu.col(i) /= area2 * area2; // normal is non-unit
			
			x.col(i) = -gu.col(i).normalized();
			//x.col(i) = -gu.col(i).stableNormalized();
		}

		Eigen::VectorXd gx = Eigen::VectorXd::Zero(nv);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto v0 = mesh.faces[i]->edge->vert->id;
			auto v1 = mesh.faces[i]->edge->next->vert->id;
			auto v2 = mesh.faces[i]->edge->next->next->vert->id;

			auto p0 = mesh.position(v0);
			auto p1 = mesh.position(v1);
			auto p2 = mesh.position(v2);

			double cd02 = mesh.cot(mesh.faces[i]->edge) *             (p0 - p2).dot(x.col(i));
			double cd10 = mesh.cot(mesh.faces[i]->edge->next) *       (p1 - p0).dot(x.col(i));
			double cd21 = mesh.cot(mesh.faces[i]->edge->next->next) * (p2 - p1).dot(x.col(i));

			gx(v0) += cd10 -  cd02;
			gx(v1) += cd21 -  cd10;
			gx(v2) += cd02 -  cd21;
		}

		gx *= 0.5;
		//gx.normalize();

		solver.compute(-lbo);
		Eigen::VectorXd dist = solver.solve(gx);
		return dist.array() - dist.minCoeff();
	}

	///
	/// auto solver1 = gp::util::heatDiffusionSolver(lbo, areas, pow(mesh.avgEdgeLength(), 2));
	/// auto solver2 = gp::util::laplaceSolver(-lbo);
	/// 
	template <typename T=size_t>
	Eigen::VectorXd geodesicHeatDist(const heds::Mesh<Eigen::MatrixX3d>& mesh,
		const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver1, 
		const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver2,
		const std::vector<T>& src) {

		size_t nv = mesh.getNumVerts(), nf = mesh.getNumFaces();

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto i : src) delta(i) = 1.0;

		Eigen::VectorXd u = solver1.solve(delta);

		Eigen::MatrixXd gu = Eigen::MatrixXd::Zero(3, nf);
		Eigen::MatrixXd x(3, nf);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto normal = mesh.normal(mesh.faces[i]);
			auto area2 = normal.norm(); // 0.5 * ||u x v||

			auto v0 = mesh.faces[i]->edge->vert->id;
			auto v1 = mesh.faces[i]->edge->next->vert->id;
			auto v2 = mesh.faces[i]->edge->next->next->vert->id;

			auto p0 = mesh.position(v0);
			auto p1 = mesh.position(v1);
			auto p2 = mesh.position(v2);

			gu.col(i) += u(v0) * (normal.cross(p2 - p1));
			gu.col(i) += u(v1) * (normal.cross(p0 - p2));
			gu.col(i) += u(v2) * (normal.cross(p1 - p0));
			gu.col(i) /= area2 * area2; // normal is non-unit

			x.col(i) = -gu.col(i).normalized();
		}

		Eigen::VectorXd gx = Eigen::VectorXd::Zero(nv);

		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto v0 = mesh.faces[i]->edge->vert->id;
			auto v1 = mesh.faces[i]->edge->next->vert->id;
			auto v2 = mesh.faces[i]->edge->next->next->vert->id;

			auto p0 = mesh.position(v0);
			auto p1 = mesh.position(v1);
			auto p2 = mesh.position(v2);

			double cd02 = mesh.cot(mesh.faces[i]->edge) * (p0 - p2).dot(x.col(i));
			double cd10 = mesh.cot(mesh.faces[i]->edge->next) * (p1 - p0).dot(x.col(i));
			double cd21 = mesh.cot(mesh.faces[i]->edge->next->next) * (p2 - p1).dot(x.col(i));

			gx(v0) += cd10 - cd02;
			gx(v1) += cd21 - cd10;
			gx(v2) += cd02 - cd21;
		}

		gx *= 0.5;

		gx = solver2.solve(gx);
		return gx.array() - gx.minCoeff();
	}

}