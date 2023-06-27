#pragma once

#include <vector>
#include <kika/surface_mesh.hpp>
#include <kika/diff_geometry.hpp>

namespace kika::gd {

// 
// unreferenced vertex as value:
// dist = (dist.array() >= std::numeric_limits<double>::max()).select(value, dist);
//
class GeodesicSolver {
protected:
	std::reference_wrapper<const heds::SurfaceMesh> mesh;
	std::vector<std::pair<double, uint32_t>> path;

public:
	GeodesicSolver(const heds::SurfaceMesh& mesh) : mesh(mesh) {}

	// get distance vector
	Eigen::VectorXd dist() const {
		Eigen::VectorXd dist(mesh.get().getNumVerts());
		for (Eigen::Index i = 0; i < dist.size(); i++) dist[i] = path[i].first;
		return dist;
	}

	// get distance value
	double value(const int i) const {
		return path[i].first;
	}

	const std::vector<std::pair<double, uint32_t>>& shortestPathTree() const {
		return path;
	}

	std::vector<uint32_t> root(uint32_t src) const {
		std::vector<uint32_t> ret;
		while (src != path[src].second) {
			ret.emplace_back(src);
			src = path[src].second;
		}
		ret.emplace_back(src);
		return ret;
	}

	std::vector<uint32_t> s2t(uint32_t src, uint32_t trg) const {
		std::vector<uint32_t> ret;
		while (src != trg && src != path[src].second) {
			ret.emplace_back(src);
			src = path[src].second;
		}
		ret.emplace_back(src);

		if (src != trg) {
			auto t2r = root(trg);
			uint32_t offset = 0;
			while (*(ret.rbegin() + offset) == *(t2r.rbegin() + offset)) offset++;

			assert(offset != 0 && "src and trg are disconnected");

			auto insert = 1 + ret.size() - offset;
			ret.resize(insert + t2r.size() - offset);
			auto itr1 = ret.begin() + insert;
			for (auto itr2 = t2r.rbegin() + offset; itr2 != t2r.rend(); ++itr1, ++itr2) *itr1 = *itr2;
		}
		return ret;
	}

};

// Dijkstra shortest path search
class DijkstraSolver : public GeodesicSolver {
	// Dijkstra search
	template <typename T = uint32_t>
	void dijkstraPath(
		const std::vector<T>& src, const double limit = std::numeric_limits<double>::max()) {

		using node = std::pair<double, uint32_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		std::vector<bool> arrived(mesh.get().getNumVerts(), false);

		for (auto i : src) {
			if (path[i].first == 0.) continue;
			path[i].first = 0.0;
			path[i].second = static_cast<uint32_t>(i);
			search.push(std::make_pair(0.0, i));
		}

		while (!search.empty()) {
			auto [d, v] = search.top();
			if (d > limit) break;
			search.pop();

			if (arrived[v]) continue;
			arrived[v] = true;

			for (const auto& u : mesh.get().verts[v]->oneringNeighbs()) {
				if (arrived[u]) continue;
				double dist = d + (mesh.get().pos.row(u) - mesh.get().pos.row(v)).norm();
				if (path[u].first > dist) {
					path[u].first = dist;
					path[u].second = v;
					search.push(std::make_pair(path[u].first, u));
				}
			}
		}
	}

	// Astar search
	template <typename T = uint32_t>
	void aStarPath(
		const std::vector<T>& src, const uint32_t trg = std::numeric_limits<uint32_t>::max(),
		const double limit = std::numeric_limits<double>::max()) {

		using node = std::tuple<double, double, uint32_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		std::vector<bool> arrived(mesh.get().getNumVerts(), false);

		Eigen::RowVector3d ptrg = mesh.get().pos.row(trg);

		for (auto i : src) {
			if (path[i].first == 0.0) continue;
			path[i].first = 0.0;
			path[i].second = static_cast<uint32_t>(i);
			if (i == trg) return;
			search.push(std::make_tuple((ptrg - mesh.get().pos.row(i)).norm(), 0.0, i));
		}

		while (!search.empty()) {
			auto [h, d, v] = search.top();
			if (v == trg || d > limit) break;
			search.pop();

			if (arrived[v]) continue;
			arrived[v] = true;

			for (const auto& u : mesh.get().verts[v]->oneringNeighbs()) {
				if (arrived[u]) continue;
				double dist = d + (mesh.get().pos.row(u) - mesh.get().pos.row(v)).norm();
				if (path[u].first > dist) {
					path[u].first = dist;
					path[u].second = v;
					search.push(std::make_tuple(dist + (mesh.get().pos.row(u) - ptrg).norm(), dist, u));
				}
			}
		}
	}

public:
	/// @brief Dijkstra algorithm (or Astar)
	/// @param mesh : surface mesh
	DijkstraSolver(const heds::SurfaceMesh& mesh) : GeodesicSolver(mesh) {}

	template <typename T = uint32_t>
	DijkstraSolver& solve(const std::vector<T>& src,
		const uint32_t trg = std::numeric_limits<uint32_t>::max(),
		const double limit = std::numeric_limits<double>::max()) {

		path.clear();
		path.resize(mesh.get().getNumVerts(), std::make_pair(std::numeric_limits<double>::max(), 0));
		if (trg < mesh.get().getNumVerts()) aStarPath(src, trg, limit);
		else dijkstraPath(src, limit);
		return *this;
	}

};

// Fast marching geodesic distance
class FastMarchingSolver : public GeodesicSolver {

	double fmdist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3,
		const double d1, const double d2) const {
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
		if (t > d && (CD < a/dotUV) && (CD > a * dotUV)) {
			dist = (d1 < d2 ? d1 : d2) + t;
		} else {
			double di = beta + d1;
			double dj = alpha + d2;
			dist = di < dj ? di : dj;
		}
		return dist;
	}

	template <typename T = uint32_t>
	void fastMarchingPath(const std::vector<T>& src,
		const uint32_t trg = std::numeric_limits<uint32_t>::max(),
		const double limit = std::numeric_limits<double>::max()) {

		using node = std::pair<double, uint32_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		path.resize(mesh.get().getNumVerts(), std::make_pair(std::numeric_limits<double>::max(), 0));
		std::vector<bool> arrived(mesh.get().getNumVerts(), false);

		for (auto i : src) {
			if (path[i].first == 0.0) continue;

			path[i].first = 0.0;
			path[i].second = static_cast<uint32_t>(i);
			arrived[i] = true;
			if (i == trg) return;

			Eigen::RowVector3d s = mesh.get().pos.row(i);
			for (const auto& v : mesh.get().verts[i]->oneringNeighbs()) {
				double d = (mesh.get().pos.row(v) - s).norm();
				if (path[v].first > d) {
					path[v].first = d;
					path[v].second = static_cast<uint32_t>(i);
					search.emplace(d, v);
				}
			}
		}

		while (!search.empty()) {
			auto [d, v] = search.top();
			if (v == trg || d > limit) break;
			search.pop();

			if (arrived[v]) continue;
			arrived[v] = true;

			for (const auto& e : mesh.get().verts[v]->incidentEdges<heds::Ptr>()) {
				if (e->face == nullptr) continue;
				auto u = e->next->vert->id;
				auto k = e->prev()->vert->id;

				if (arrived[u] == arrived[k]) continue;
				if (arrived[k]) std::swap(u, k);

				double dist = fmdist(
					mesh.get().pos.row(v),
					mesh.get().pos.row(u),
					mesh.get().pos.row(k),
					path[v].first, path[u].first);

				if (path[k].first > dist) {
					path[k].first = dist;
					path[k].second = u;
					search.emplace(dist, k);
				}
			}
		}
	}

	#if false
	// TODO : bug
	template <typename T = uint32_t>
	std::vector<std::pair<double, uint32_t>> aStarPath(const std::vector<T>& src,
		const uint32_t trg = std::numeric_limits<uint32_t>::max(),
		const double limit = std::numeric_limits<double>::max()) const {
	
		using node = std::tuple<double, double, uint32_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		std::vector<std::pair<double, uint32_t>> nodes(mesh.get().getNumVerts(), std::make_pair(std::numeric_limits<double>::max(), 0));
		std::vector<bool> arrived(mesh.get().getNumVerts(), false);
	
		Eigen::RowVector3d ptrg = mesh.get().pos.row(trg);
	
		for (auto i : src) {
			if (nodes[i].first == 0.0) continue;
	
			nodes[i].first = 0.0;
			nodes[i].second = static_cast<uint32_t>(i);
			arrived[i] = true;
	
			if (i == trg) return nodes;
	
			Eigen::RowVector3d s = mesh.get().pos.row(i);
			for (const auto& v : mesh.get().verts[i]->oneringNeighbs()) {
				double dist = (s - mesh.get().pos.row(v)).norm();
				if (nodes[v].first > dist) {
					nodes[v].first = dist;
					nodes[v].second = static_cast<uint32_t>(i);
					search.emplace(dist + (ptrg - mesh.get().pos.row(v)).norm(), dist, v);
				}
			}
		}
	
		while (!search.empty()) {
			auto [h, d, v] = search.top();
			if (v == trg || d > limit) break;
			search.pop();
	
			if (arrived[v]) continue;
			arrived[v] = true;
	
			for (const auto& e : mesh.get().verts[v]->incidentEdges<heds::Ptr>()) {
				if (e->isBoundary()) continue;
				auto u = e->next->vert->id;
				auto k = e->prev()->vert->id;
	
				if (arrived[u] == arrived[k]) continue;
				if (arrived[k]) std::swap(u, k);
	
				double dist = fmdist(
					mesh.get().pos.row(v),
					mesh.get().pos.row(u),
					mesh.get().pos.row(k),
					nodes[v].first, nodes[u].first);
	
				if (nodes[k].first > dist) {
					nodes[k].first = dist;
					nodes[k].second = u;
					double heur = (ptrg - mesh.get().pos.row(k)).norm();
					search.emplace(dist + heur, dist, k);
				}
			}
		}
		return nodes;
	}
	#endif

public:

	// fast marching geodesic distance
	FastMarchingSolver(const heds::SurfaceMesh& mesh) : GeodesicSolver(mesh) {
	}

	template <typename T = uint32_t>
	FastMarchingSolver& solve(const std::vector<T>& src,
		const uint32_t trg = std::numeric_limits<uint32_t>::max(),
		const double limit = std::numeric_limits<double>::max()) {

		path.clear();
		path.resize(mesh.get().getNumVerts(), std::make_pair(std::numeric_limits<double>::max(), 0));
		//if (trg < mesh.get().getNumVerts()) return aStarPath(src, trg, limit);
		fastMarchingPath(src, trg, limit);
		return *this;
	}

};

// Virtual source propagation (triangle mesh)
class VirtualSourceSolver {
	std::reference_wrapper<const heds::SurfaceMesh> mesh;

	std::vector<double> face2centerDist;
	std::vector<double> face2extraDist;
	std::vector<double> edgeLength;
	std::vector<double> edge2vertexDist;
	std::vector<bool> behindMask;
	double ael;
	Eigen::VectorXd distance;

	void computeAel() {
		int ne = (int)mesh.get().getNumEdges();
		edgeLength.resize(ne);
		ael = 0.0;
		double _ael = 0.0;
		#pragma omp parallel for reduction(+:_ael)
		for (int i = 0; i < ne; i++) {
			double length = mesh.get().length(mesh.get().edges[i]);
			edgeLength[mesh.get().edges[i]->uid()] = length;
			_ael += length;
		}
		ael = _ael / (double)ne;
	}

	template <typename T = uint32_t>
	void init(const std::vector<T>& src, std::vector<heds::pte>& searchMain) {
		std::vector<bool> arrived(mesh.get().getNumVerts(), false);

		for (auto s : src) {
			if (arrived[s]) continue;
			for (const auto& e2 : mesh.get().verts[s]->incidentEdges<heds::Ptr>()) {
				if (e2->face == nullptr) continue;

				auto& e0 = e2->next;
				auto& e1 = e0->next;

				Eigen::RowVector3d p0 = mesh.get().pos.row(e0->vert->id);
				Eigen::RowVector3d p1 = mesh.get().pos.row(e1->vert->id);
				Eigen::RowVector3d p2 = mesh.get().pos.row(e2->vert->id);

				Eigen::RowVector3d centroid = (p0 + p1 + p2) / 3.0;

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
				searchMain.emplace_back(e1);
			}
			arrived[s] = true;
		}
	}

	void computeVwiseDist() {
		int nv = (int)mesh.get().getNumVerts();
		distance.resize(nv);

		#pragma omp parallel for
		for (int i = 0; i < nv; i++) {
			double mindist = std::numeric_limits<double>::max();

			for (const auto& e : mesh.get().verts[i]->incidentEdges<heds::Ptr>()) {
				if (e->face == nullptr) continue;
				double d = edge2vertexDist[e->id];

				mindist = std::min(mindist, face2extraDist[e->face->id] + sqrt(d));
			}

			distance(i) = mindist;
		}
	}

public:

	// Virtual source propagation (triangle mesh)
	VirtualSourceSolver(const heds::SurfaceMesh& mesh) : mesh(mesh) {
		computeAel();
	}

	template <typename T = uint32_t>
	VirtualSourceSolver& solve(const std::vector<T>& src, const bool resetAel = false) {
		constexpr double THRESHOLD_c = 5.1424;
		constexpr double THRESHOLD_g = 4.20638;
		constexpr double THRESHOLD_h = 0.504201;
		constexpr double THRESHOLD_hg = 2.84918;
		constexpr double LAMBDA[16] ={
			0.320991, 0.446887, 0.595879,  0.270094,
			0.236679, 0.159685,  0.0872932, 0.434132,
			1.0, 0.726262, 0.0635997, 0.0515979,
			0.56903, 0.0447586, 0.0612103, 0.718198 };

		if (resetAel) computeAel();
		int nf = (int)mesh.get().getNumFaces(), ne = (int)mesh.get().getNumEdges();
		face2centerDist.clear();
		face2centerDist.resize(nf, std::numeric_limits<double>::max());
		face2extraDist.clear();
		face2extraDist.resize(nf, 0.);
		edge2vertexDist.clear();
		edge2vertexDist.resize(2*ne, std::numeric_limits<double>::max());
		behindMask.clear();
		behindMask.resize(2*ne, false);

		std::vector<heds::pte> searchMain, searchSub;
		init(src, searchMain);

		double iteration = 0.0;
		while (!searchMain.empty()) {
			iteration += ael;
			searchSub.clear();

			for (size_t i = 0; i < searchMain.size(); i++) {
				auto& srcEdge = searchMain[i];
				auto& trgEdge = srcEdge->pair;

				if (trgEdge->face == nullptr) continue;

				auto& trgEdgeNext = trgEdge->next;
				auto& trgEdgePrev = trgEdgeNext->next; //(prev)

				auto e1 = edgeLength[trgEdge->uid()];
				auto e2 = edgeLength[trgEdgePrev->uid()];
				auto e3 = edgeLength[trgEdgeNext->uid()];

				double& refDist = face2centerDist[trgEdge->face->id];

				bool isBehind = behindMask[srcEdge->id];
				auto d1 = edge2vertexDist[srcEdge->id];
				auto d2 = edge2vertexDist[srcEdge->next->next->id]; //(prev)
				auto sigmaT = face2extraDist[srcEdge->face->id];

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

				double dist;
				double dA, dB, dC;

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
					double maxe = std::max(e1, std::max(e2, e3));
					double mine = std::min(e1, std::min(e2, e3));
					int mask =
						(maxe > THRESHOLD_c * e1) +	(maxe > THRESHOLD_g * mine) * 2 +
						(py < THRESHOLD_h* e1) * 4 +	(py < THRESHOLD_hg* maxe) * 8;
					double qx = px * (1.0 - LAMBDA[mask]) + cx * LAMBDA[mask];
					double qy = py * (1.0 - LAMBDA[mask]) + cy * LAMBDA[mask];

					double ttx = qx * sy + sx * qy;

					if (ttx < 0) {
						dA = 0.0;
						dB = e1 * e1;
						dC = e2 * e2;

						sigmaT += ds1;
						dist = sigmaT + dc1;
					} else if (ttx > e1 * (qy + sy)) {
						dA = e1 * e1;
						dB = 0.0;
						dC = e3 * e3;

						sigmaT += ds2;
						dist = sigmaT + dc2;
					} else {
						ttx = px * sy + sx * py;
						dA = d1;
						dB = d2;
						dC = util::pow2(px - sx) + util::pow2(py + sy);

						dist = sigmaT + sqrt(util::pow2(cx - sx) + util::pow2(cy + sy));

						behindMask[trgEdgePrev->id] = (ttx < 0);
						behindMask[trgEdgeNext->id] = (ttx > e1 * (py + sy));
					}
				}

				if (dist < refDist) {
					refDist = dist;

					face2extraDist[trgEdge->face->id] = sigmaT;
					edge2vertexDist[trgEdge->id] = dB;
					edge2vertexDist[trgEdgePrev->id] = dA;
					edge2vertexDist[trgEdgeNext->id] = dC;

					auto& searchNext = dist < iteration ? searchMain : searchSub;
					searchNext.emplace_back(trgEdgePrev);
					searchNext.emplace_back(trgEdgeNext);
				}
			}

			std::swap(searchMain, searchSub);
		}

		computeVwiseDist();
		return *this;
	}

	const Eigen::VectorXd& dist() const {
		return distance;
	}

	double value(const int i) const {
		return distance[i];
	}

};

class GeodesicHeatSolver {
	std::reference_wrapper<heds::SurfaceMesh> mesh;
	std::reference_wrapper<diff::DiscreteLaplacian> dsl;
	Eigen::VectorXd distance;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> heat, poisson;

public:

	/// @param lambda : time step
	GeodesicHeatSolver(heds::SurfaceMesh& mesh, diff::DiscreteLaplacian& dsl, double lambda = 0) : mesh(mesh), dsl(dsl) {
		if (lambda == 0) lambda = util::pow2(mesh.avgEdgeLength());

		if (dsl.lbo.rows() == 0) dsl.laplacianOp();
		if (dsl.s0.rows() == 0) dsl.vertexAreas();
		if (dsl.grad[0].rows() == 0) dsl.gradientOp();
		if (dsl.div[0].rows() == 0) dsl.divergenceOp();

		heat.compute(eig::spdiag(dsl.s0) + lambda * dsl.lbo);
		poisson.compute(-dsl.lbo);
	}

	template <typename T = uint32_t>
	GeodesicHeatSolver& solve(const std::vector<T>& src) {
		size_t nv = mesh.get().getNumVerts(), nf = mesh.get().getNumFaces();

		Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
		for (auto i : src) delta(i) = 1.0;
		
		Eigen::VectorXd u = heat.solve(delta);

		Eigen::MatrixX3d x = - dsl.get().integratedGradient(u);

		Eigen::VectorXd dx = dsl.get().div * x;

		distance = poisson.solve(-dx);
		distance.array() -= distance.minCoeff();
		return *this;
	}

	const Eigen::VectorXd& dist() const {
		return distance;
	}

	double value(const int i) const {
		return distance[i];
	}
};

inline void test() {
	//uint32_t target = 10000;
	//target = (target >= mesh.getNumVerts()) ? (uint32_t)mesh.getNumVerts() - 1 : target;
	//{
	//	kika::util::Timer timer;
	//	auto dist = kika::gd::DijkstraSolver(mesh).solve({ 0 }).dist();
	//	std::cout << dist[target] << std::endl;
	//}
	//
	//{
	//  kika::util::Timer timer;
	//  auto dist = kika::gd::FastMarchingSolver(mesh).solve({ 0 }).dist();
	//  std::cout << dist[target] << std::endl;
	//}
	//
	//{
	//  kika::util::Timer timer;
	//  auto dist = kika::gd::VirtualSourceSolver(mesh).solve({ 0 }).dist();
	//  std::cout << dist[target] << std::endl;
	//}
	//{
	//	kika::util::Timer timer;
	//	auto dist = kika::gd::DijkstraSolver(mesh).solve({ 0 }, target).dist();
	//	std::cout << dist[target] << std::endl;
	//}
	//
	//{
	//	kika::util::Timer timer;
	//	auto dist = kika::gd::FastMarchingSolver(mesh).solve({ 0 }).dist();
	//	std::cout << dist[target] << std::endl;
	//}
	//
	//{
	//	kika::util::Timer timer;
	//	auto dist = kika::gd::FastMarchingSolver(mesh).solve({ 0 }, target).dist();
	//	std::cout << dist[target] << std::endl;
	//}
	//
	//{
	//	kika::util::Timer timer;
	//	auto dist = kika::gd::VirtualSourceSolver(mesh).solve({ 0 }).dist();
	//	std::cout << dist[target] << std::endl;
	//}
}

}