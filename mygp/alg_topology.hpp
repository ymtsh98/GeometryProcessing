#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include <mygp/geoutil.hpp>
#include <mygp/union_find.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/distance.hpp>
#include <mygp/eigutil.hpp>

namespace gp::topo::impl {
inline std::vector<size_t> getPathToSource(const std::vector<std::pair<size_t, double>>& shortestPathTree, const size_t from) {
	std::vector<size_t> path ={ from };
	size_t i = from;
	while (i != shortestPathTree[i].first) {
		i = shortestPathTree[i].first;
		path.emplace_back(i);
	}
	return path;
}

inline std::pair<std::vector<size_t>, double> getPathDistToSource(const std::vector<std::pair<size_t, double>>& shortestPathTree, const size_t from) {
	std::vector<size_t> path ={ from };
	double dist = 0.0;
	size_t i = from;
	while (i != shortestPathTree[i].first) {
		dist += shortestPathTree[i].second;
		i = shortestPathTree[i].first;
		path.emplace_back(i);
	}
	return std::make_pair(path, dist);
}

// check the number of genus on separated mesh (there is one boundary at least)
template <typename T>
int countNumGenus(const heds::Mesh<T>& mesh, const std::vector<bool>& mask) {
	int v = 0, e = 0, f = 0, b = 1; // b = 1 ? 

	for (size_t i = 0; i < mask.size(); i++) v += mask[i];

	for (const auto& face : mesh.faces) {
		bool flag = true;
		for (const auto& vert : face->adjacentVindex()) {
			if (!mask[vert]) {
				flag = false;
				break;
			}
		}

		if (flag) f++;
	}

	for (const auto& edge : mesh.edges)
		if (mask[edge->vert->id] && mask[edge->pair->vert->id]) e++;

	return -((v - e + f) - 2 + b) / 2;
}

template <typename T>
bool isNonSeparation(const heds::Mesh<T>& mesh, const std::vector<size_t>& cycle, const std::unordered_set<size_t>& cycleIdx) {

	// check cycle has interior points
	size_t g1{}, g2{};
	size_t count = 0;
	for (size_t i = 0; i < cycle.size(); i++) {
		size_t s = cycle[i], t = cycle[(i+1)%cycle.size()];

		auto edge = mesh.halfedge((uint32_t)s, (uint32_t)t);
		if (cycleIdx.contains(edge->next->vert->id) || cycleIdx.contains(edge->pair->next->vert->id)) {
			count++;
		} else {
			g1 = edge->next->vert->id;
			g2 = edge->pair->next->vert->id;
		}
	}
	if (count == cycle.size()) return false;

	// check inside and outside surfaces doesn't separeted

	auto edge = mesh.halfedge((uint32_t)cycle[0], (uint32_t)cycle[1]);
	std::vector<size_t> queue ={ g1, g2 };
	std::vector<int> surfaceGroup(mesh.getNumVerts(), 0);
	surfaceGroup[g1] = 1;
	surfaceGroup[g2] = 2;

	while (!queue.empty()) {
		auto i = queue.back();
		queue.pop_back();

		for (const auto& v : mesh.verts[i]->oneringVindex()) {
			if (surfaceGroup[v] != 0 && (surfaceGroup[v] != surfaceGroup[i])) return true;
			if (surfaceGroup[v] == 0 && !cycleIdx.contains(v)) {
				surfaceGroup[v] = surfaceGroup[i];
				queue.emplace_back(v);
			}
		}
	}

	return false;
}

template <typename T>
bool isNonContraction(const heds::Mesh<T>& mesh, const std::vector<size_t>& cycle, const std::unordered_set<size_t>& cycleIdx) {

	// check cycle has interior points
	size_t g1{}, g2{};
	size_t count = 0;
	for (size_t i = 0; i < cycle.size(); i++) {
		size_t s = cycle[i], t = cycle[(i+1)%cycle.size()];

		auto edge = mesh.halfedge((uint32_t)s, (uint32_t)t);
		if (cycleIdx.contains(edge->next->vert->id) || cycleIdx.contains(edge->pair->next->vert->id)) {
			count++;
		} else {
			g1 = edge->next->vert->id;
			g2 = edge->pair->next->vert->id;
		}
	}
	if (count == cycle.size()) return false;


	// check inside and outside surfaces doesn't separeted
	auto edge = mesh.halfedge((uint32_t)cycle[0], (uint32_t)cycle[1]);
	std::vector<size_t> queue ={ g1, g2 };
	std::vector<int> surfaceGroup(mesh.getNumVerts(), 0);
	surfaceGroup[g1] = 1;
	surfaceGroup[g2] = 2;

	while (!queue.empty()) {
		auto i = queue.back();
		queue.pop_back();

		for (const auto& v : mesh.verts[i]->oneringVindex()) {
			if (surfaceGroup[v] != 0 && (surfaceGroup[v] != surfaceGroup[i])) return true;
			if (surfaceGroup[v] == 0 && !cycleIdx.contains(v)) {
				surfaceGroup[v] = surfaceGroup[i];
				queue.emplace_back(v);
			}
		}
	}

	// check separated surfaces are non zero genus

	std::vector<bool> mask(mesh.getNumVerts());
	for (size_t i = 0; i < mask.size(); i++) mask[i] = (surfaceGroup[i] == 1 ? true : false);
	int sg1 = countNumGenus(mesh, mask);

	for (size_t i = 0; i < mask.size(); i++) mask[i] = (surfaceGroup[i] == 2 ? true : false);
	int sg2 = countNumGenus(mesh, mask);

	if ((sg1 == 0 && sg2 != 0) ||
		(sg1 != 0 && sg2 == 0)) return false;

	return true;
}

inline void gaussJordanElimZ2(Eigen::Matrix<bool, -1, -1>& mat, const bool pb = false) {
	auto rows = mat.rows(), cols = mat.cols();

	if (pb) std::cout << "gauss-jordan" << std::endl;

	for (Eigen::Index i = 0, j = 0; (i < rows) && (j < cols); j++) {
		if (pb) util::printProgress(std::max((double)i / rows, (double)j/cols));
		
		Eigen::Index maxRow{}, ignore;
		auto val = mat.block(i, j, rows-i, 1).maxCoeff(&maxRow, &ignore);
		if (val == 0) continue;		
		mat.row(i).swap(mat.row(maxRow+i));

		Eigen::Matrix<bool, 1, -1>&& row = mat.block(i, j+1, 1, cols-j-1);
		for (Eigen::Index k = i+1; k < rows; k++) {
			if (mat(k, j)) {
				#pragma warning(push)
				#pragma warning(disable:4804)
				mat.block(k, j+1, 1, cols-j-1) -= row;
				#pragma warning(pop)
			}
		}

		i++;
	}
	if (pb) std::cout << std::endl;
}

inline std::vector<std::vector<size_t>> cycleExtraction(const std::vector<std::vector<size_t>>& adjacentList, const std::vector<std::pair<size_t, size_t>>& bridges) {
	size_t n = adjacentList.size();

	std::vector<std::vector<size_t>> cycles;
	for (size_t i = 0; i < bridges.size(); i++) {
		auto [s, t] = bridges[i];

		std::vector<size_t> parent(n);
		std::vector<size_t> queue; queue.reserve(n);
		std::vector<bool> used(n, false);

		parent[s] = s;
		queue.emplace_back(s);
		used[s] = true;

		while (!queue.empty()) {
			auto v = queue.back();
			if (v == t) break;
			queue.pop_back();

			for (const auto u : adjacentList[v]) {
				if (used[u]) continue;
				parent[u] = v;
				used[u] = true;
				queue.emplace_back(u);
			}
		}

		size_t c = t;
		std::vector<size_t> cycle;
		while (c != parent[c]) {
			cycle.emplace_back(c);
			c = parent[c];
		}
		cycle.emplace_back(c);
		cycles.emplace_back(cycle);
	}

	return cycles;
}

}

namespace gp::topo {
enum class MinCycleOpt { All, NonSeparation, NonContraction };

// compute minimum cycle candidates (opt : all, non-separation, non-contraction).
// 
// WARNING : this code does not concerned about a mesh such that has boundary or disconnected graph.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
template <typename T>
std::vector<std::vector<size_t>> minimumCycleCandidates(
	const heds::Mesh<T>& mesh, const MinCycleOpt option = MinCycleOpt::All, const bool pb = false) {
	std::vector<std::vector<size_t>> cycles;
	cycles.reserve(mesh.getNumEdges());

	if (pb) std::cout << "compute candidates" << std::endl;

	std::vector<std::vector<size_t>> paths(mesh.getNumVerts());
	std::unordered_set<size_t> cycleIdx;
	cycleIdx.reserve(2 * mesh.getNumVerts());

	for (size_t source = 0; source < mesh.getNumVerts();) {
		auto shortestPathTree = gp::dist::dijkstraPath(mesh, { source });

		for (size_t i = 0; i < paths.size(); i++)
			paths[i] = impl::getPathToSource(shortestPathTree, i);

		for (const auto& edge : mesh.edges) {
			auto [u, v] = edge->endpointIndex();

			if (u == source || v == source) continue;

			auto path1 = paths[u];
			auto path2 = paths[v];

			//check cycle is horton's cycle
			if (abs((int)path1.size() - (int)path2.size()) > 1) continue;

			cycleIdx.clear();
			cycleIdx.insert(path1.begin(), path1.end());
			cycleIdx.insert(path2.begin(), path2.end());

			//check two paths are internally disjoint
			if (cycleIdx.size() != (path1.size() + path2.size() - 1)) continue;

			cycles.emplace_back(path2);
			cycles.back().reserve(path1.size() + path2.size() - 1);
			cycles.back().insert(cycles.back().end(), path1.rbegin()+1, path1.rend());

			// check cycle in C

			if (option == MinCycleOpt::NonSeparation &&
				!impl::isNonSeparation(mesh, cycles.back(), cycleIdx)) cycles.pop_back();

			if (option == MinCycleOpt::NonContraction &&
				!impl::isNonContraction(mesh, cycles.back(), cycleIdx)) cycles.pop_back();
		}

		source++;
		if (pb) util::printProgress((double)source / mesh.getNumVerts());
	}
	if (pb) std::cout << std::endl;

	return cycles;
}

// sort cycles by edge length.
template <typename T>
void sortCycleCandidates(std::vector<std::vector<size_t>>& cycles, const heds::Mesh<T>& mesh) {
	std::vector<int> sortIndex(cycles.size());
	{
		std::vector<double> cycleLengths(cycles.size(), 0.0);
		for (size_t i = 0; i < cycleLengths.size(); i++) {
			for (size_t j = 0, k = cycles[i].size(); j < k; j++) {
				cycleLengths[i] += mesh.length(
					mesh.halfedge((uint32_t)cycles[i][j], (uint32_t)cycles[i][(j+1)%k]));
			}
		}

		std::iota(sortIndex.begin(), sortIndex.end(), 0);
		std::sort(sortIndex.begin(), sortIndex.end(),
			[cycleLengths](auto& l, auto& r) { return cycleLengths[l] < cycleLengths[r]; });
	}

	auto copyCycles = cycles;
	for (size_t i = 0; i < cycles.size(); i++) {
		cycles[i] = copyCycles[sortIndex[i]];
	}
}

// compute minimum cycle basis.
// 
// WARNING : this code is not concerned about a mesh such that has boundary or disconnected graph.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
template <typename T>
std::vector<std::vector<size_t>> minimumCycleBasis(
	const heds::Mesh<T>& mesh, const MinCycleOpt option = MinCycleOpt::All, const bool pb = false) {
	auto cycles = minimumCycleCandidates(mesh, option, pb);
	sortCycleCandidates(cycles, mesh);

	Eigen::Matrix<bool, -1, -1> E= 
		Eigen::Matrix<bool, -1, -1>::Zero(mesh.getNumEdges(), cycles.size());
	for (size_t i = 0; i < cycles.size(); i++) {
		for (size_t j = 0, k = cycles[i].size(); j < k; j++) {
			uint32_t u = (uint32_t)cycles[i][j], v = (uint32_t)cycles[i][(j+1)%k];
			E(mesh.halfedge(u, v)->id/2, i) = true;
		}
	}
	
	impl::gaussJordanElimZ2(E, pb);

	std::vector<std::vector<size_t>> basis;
	for (size_t i = 0, j = 0; (i < cycles.size()) && (j < mesh.getNumEdges()); i++) {
		if (E(j, i)) {
			basis.emplace_back(cycles[i]);
			j++;
		}
	}

	return basis;
}

// compute minimum spanning tree (kruskal algorithm).
// 
// @param mesh
// @param weights : sort weights ({} no sort)
// @param mask    : use edge mask ({} all edge evaluate)
template <typename T1, typename T2 = double, typename T3 = bool>
std::vector<size_t> kruskalSpanningTree(const heds::Mesh<T1>& mesh, const std::vector<T2>& weights={}, 
	const std::vector<T3>& mask={}) {
	size_t n = mesh.getNumVerts();

	util::UnionFind uf(n);

	std::vector<size_t> index;
	if (mask.size()) {
		index.reserve(mesh.getNumEdges());
		for (size_t i = 0; i < mesh.getNumEdges(); i++) if (mask[i]) index.emplace_back(i);
	} else {
		index.resize(mesh.edges.size());
		std::iota(index.begin(), index.end(), 0);
	}

	if (weights.size()) {
		std::sort(index.begin(), index.end(), [weights](auto& l, auto& r) { return weights[l] < weights [r] ; });
	}

	std::vector<size_t> mst;
	mst.reserve(n-1);

	for (const auto i : index) {
		auto [u, v] = mesh.edges[i]->endpointIndex();

		if (!uf.same(u, v)) {
			uf.unite(u, v);
			mst.emplace_back(i);
			if (mst.size() == n - 1) break;
		}
	}

	return mst;
}

// compute tree cotree decomposition.
// 
// WARNING : (primal cycle version) this code does not concerned about a mesh such that has boundary or disconnected graph.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
// @boundary : include boundary cycles (return dual cycle)
template <typename T>
std::vector<std::vector<size_t>> treeCotreeDecomp(const heds::Mesh<T>& mesh, 
	const bool dualCycle = false, const bool boundary = false) {
	if (boundary == true && dualCycle == false) 
		throw std::runtime_error("Tree-Cotree : boundary cycle can only return cycle ver.");

	//---------- prepare primal mst ----------
	
	std::vector<bool> bmask(mesh.getNumEdges(), true);
	if (dualCycle && boundary) 
		for (const auto& e : mesh.edges) 
		if (e->vert->isBoundary() || e->pair->vert->isBoundary()) bmask[e->id/2] = false;

	auto primMst = kruskalSpanningTree(mesh, {}, bmask);

	std::vector<bool> inPrimMst(mesh.getNumEdges(), false);
	for (const auto i : primMst) inPrimMst[i] = true;

	//---------- construct dual mesh ----------
	auto [dverts, dfaces] = mesh.dual();
	heds::Mesh<T> dmesh(dverts, dfaces);

	//---------- compute cotree ----------
	std::vector<bool> cotree(dmesh.getNumEdges(), false);

	for (const auto& edge : mesh.edges) {
		if (inPrimMst[edge->id/2]) continue;
		if (edge->isBoundary()) continue;

		auto e = dmesh.halfedge(edge->adjacentFindex());
		cotree[e->id / 2] = true;
	}

	//---------- compute dual mst ----------
	auto dualMst = kruskalSpanningTree(dmesh, {}, cotree);
	std::vector<bool> inDualMst(dmesh.getNumEdges(), false);
	for (const auto i : dualMst) inDualMst[i] = true;

	//---------- find bridges (neither primal mst nor cotree mst) ----------
	std::vector<std::pair<size_t, size_t>> bridges;
	for (const auto& edge : dmesh.edges) {
		if (!cotree[edge->id/2]) continue;
		if (inDualMst[edge->id/2]) continue;

		if (dualCycle) bridges.emplace_back(edge->pair->endpointIndex());
		else bridges.emplace_back(mesh.halfedge(edge->adjacentFindex())->endpointIndex());
	}

	//---------- cycle extraction ----------
	std::vector<std::vector<size_t>> adList(mesh.getNumVerts());
	if (dualCycle) {
		adList.resize(dmesh.getNumVerts());
		for (size_t i = 0; i < dualMst.size(); i++) {
			auto [u, v] = dmesh.edges[dualMst[i]]->endpointIndex();
			adList[u].emplace_back(v);
			adList[v].emplace_back(u);
		}
	} else {
		for (size_t i = 0; i < primMst.size(); i++) {
			auto [u, v] = mesh.edges[primMst[i]]->endpointIndex();
			adList[u].emplace_back(v);
			adList[v].emplace_back(u);
		}
	}

	auto cycles = impl::cycleExtraction(adList, bridges);
	return cycles;
}

// compute minimum homology basis.
// 
// WARNING : this code does not concerned about a mesh such that has boundary or disconnected graph.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
template <typename T>
std::vector<std::vector<size_t>> minimumHomologyBasis(
	const heds::Mesh<T>& mesh, const bool pb = false) {
	//--------------- prepare resource -----------------
	auto [dverts, dfaces] = mesh.dual();
	heds::Mesh<T> dmesh(dverts, dfaces);

	std::vector<std::vector<size_t>> cycles;
	cycles.reserve(mesh.getNumVerts() * 2);

	{
		if (pb) std::cout << "compute homology basis candidates" << std::endl;

		std::vector<std::vector<size_t>> paths(mesh.getNumVerts());
		std::vector<double> pathLengths(mesh.getNumVerts());
		std::vector<bool> inSpt(mesh.getNumEdges(), false);

		std::vector<double> loopLength(dmesh.getNumEdges(), std::numeric_limits<double>::max());

		std::vector<bool> inMst(dmesh.getNumEdges(), false);

		for (size_t source = 0; source < mesh.getNumVerts();) {
			//---------- compute shortest path tree ----------
			auto shortestPathTree = gp::dist::dijkstraPath(mesh, { source });

			std::fill(inSpt.begin(), inSpt.end(), false);

			for (size_t i = 0; i < paths.size(); i++) {
				std::tie(paths[i], pathLengths[i]) = impl::getPathDistToSource(shortestPathTree, i);
				if (i == source) continue;
				uint32_t u = (uint32_t)shortestPathTree[i].first, v = (uint32_t)i;
				auto edge = mesh.halfedge(u, v);
				inSpt[edge->id/2] = true;
			}

			//---------- enumerate cycle candidates (greedy tree-cotree decomposition) ----------
			std::fill(loopLength.begin(), loopLength.end(), std::numeric_limits<double>::max());
			std::vector<bool> cotree(dmesh.getNumEdges(), false);

			for (const auto& edge : mesh.edges) {
				if (inSpt[edge->id/2]) continue;
				if (edge->isBoundary()) continue;

				auto [u, v] = edge->endpointIndex();
				auto de = dmesh.halfedge(edge->adjacentFindex());

				//loopLength[edge->id/2] = -((int)paths[u].size() + (int)paths[v].size() - 1);
				loopLength[de->id/2] = -(pathLengths[u] + pathLengths[v]);
				cotree[de->id/2] = true;
			}

			auto maxSpanningTree = kruskalSpanningTree(dmesh, loopLength, cotree);
			std::fill(inMst.begin(), inMst.end(), false);
			for (const auto i : maxSpanningTree) inMst[i] = true;

			std::vector<std::pair<size_t, size_t>> bridges;
			for (const auto& edge : dmesh.edges) {
				if (!cotree[edge->id/2]) continue;
				if (inMst[edge->id/2]) continue;
				if (edge->isBoundary()) continue;
				bridges.emplace_back(mesh.halfedge(edge->adjacentFindex())->endpointIndex());
			}
			
			std::vector<std::vector<size_t>> adList(mesh.getNumVerts());
			for (size_t i = 0; i < inSpt.size(); i++) {
				if (!inSpt[i]) continue;
				auto [u, v] = mesh.edges[i]->endpointIndex();
				adList[u].emplace_back(v);
				adList[v].emplace_back(u);
			}
			
			auto tct = impl::cycleExtraction(adList, bridges);
			cycles.insert(cycles.end(), tct.begin(), tct.end());

			source++;
			if (pb) util::printProgress((double)source / mesh.getNumVerts());
		}

		if (pb) std::cout << std::endl;
	}

	sortCycleCandidates(cycles, mesh);

	//--------------- test intersection (homology signature) -----------------

	auto dualCycles = treeCotreeDecomp(dmesh);
	
	std::vector<std::vector<bool>> inDualCycles(dualCycles.size(), 
		std::vector<bool>(dmesh.getNumEdges(), false));
	for (size_t i = 0; const auto& cycle : dualCycles) {
		for (size_t j = 0, k = cycle.size(); j < k; j++) {
			uint32_t u = (uint32_t)cycle[j], v = (uint32_t)cycle[(j+1)%k];
			inDualCycles[i][dmesh.halfedge(u, v)->id/2] = true;
		}
		i++;
	}
	
	Eigen::Matrix<bool, -1, -1> E = 
		Eigen::Matrix<bool, -1, -1>::Zero(dualCycles.size(), cycles.size());
	
	for (size_t i = 0; i < cycles.size(); i++) {
		for (size_t j = 0, k = cycles[i].size(); j < k; j++) {
			uint32_t u = (uint32_t)cycles[i][j], v = (uint32_t)cycles[i][(j+1)%k];
			
			auto ei = dmesh.halfedge(mesh.halfedge(u, v)->adjacentFindex())->id/2;
			for (size_t g = 0; g < dualCycles.size(); g++) {
				if (inDualCycles[g][ei]) E(g, i) = !E(g, i);
			}
		}
	}
	
	impl::gaussJordanElimZ2(E, pb);
	
	std::vector<std::vector<size_t>> basis;
	for (size_t i = 0, j = 0; (i < cycles.size()) && (j < dualCycles.size()); i++) {
		if (E(j, i)) {
			basis.emplace_back(cycles[i]);
			j++;
		}
	}

	return basis;
}

}