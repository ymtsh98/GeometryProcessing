#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include <kika/union_find.hpp>
#include <kika/halfedge_mesh.hpp>
#include <kika/distance.hpp>

namespace kika::topo::impl {

inline std::vector<uint32_t> traceCycle(
	const std::vector<std::vector<uint32_t>>& adjacentList, 
	const std::pair<uint32_t, uint32_t>& bridge) {
	size_t n = adjacentList.size();

	auto [s, t] = bridge;

	std::vector<uint32_t> parent(n);
	std::vector<uint32_t> queue;
	queue.reserve(n);
	std::vector<bool> arrived(n, false);

	parent[s] = s;
	queue.emplace_back(s);
	arrived[s] = true;

	while (!queue.empty()) {
		auto v = queue.back();
		if (v == t) break;
		queue.pop_back();

		for (const auto u : adjacentList[v]) {
			if (arrived[u]) continue;
			parent[u] = v;
			arrived[u] = true;
			queue.emplace_back(u);
		}
	}

	std::vector<uint32_t> cycle;

	uint32_t c = t;
	while (c != parent[c]) {
		cycle.emplace_back(c);
		c = parent[c];
	}
	cycle.emplace_back(c);

	return cycle;
}

inline std::vector<std::vector<uint32_t>> cyclesExtraction(
	const std::vector<std::vector<uint32_t>>& adjacentList, 
	const std::vector<std::pair<uint32_t, uint32_t>>& bridges) {

	std::vector<std::vector<uint32_t>> cycles(bridges.size());
	for (size_t i = 0; i < cycles.size(); i++) {
		cycles[i] = traceCycle(adjacentList, bridges[i]);
	}
	return cycles;
}

inline void gaussJordanElimZ2(Eigen::MatrixX<bool>& mat, const bool pb = false) {
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
}


namespace kika::topo {

// compute spanning tree (kruskal algorithm).
// 
// @param mesh
// @param weights : edge weight ({} uniform weight)
// @param mask    : edge mask ({} all edge will be evaluated)
template <typename T1 = double, typename T2 = bool>
std::vector<uint32_t> kruskalSpanningTree(const kika::heds::HalfedgeMesh& mesh, 
	const std::vector<T1>& weights={},
	const std::vector<T2>& mask={}) {
	auto n = mesh.getNumVerts();

	util::UnionFind uf(n);

	std::vector<uint32_t> index;
	if (mask.size()) {
		index.reserve(mesh.getNumEdges());
		for (uint32_t i = 0; i < mesh.getNumEdges(); i++) if (mask[i]) index.emplace_back(i);
	} else {
		index.resize(mesh.edges.size());
		std::iota(index.begin(), index.end(), 0);
	}

	if (weights.size()) {
		std::sort(index.begin(), index.end(), [weights](auto& l, auto& r) { return weights[l] < weights[r]; });
	}

	std::vector<uint32_t> mst;
	mst.reserve(n-1);

	for (const auto i : index) {
		auto [u, v] = mesh.edges[i]->endpoint();

		if (!uf.same(u, v)) {
			uf.unite(u, v);
			mst.emplace_back(i);
			if (mst.size() == n - 1) break;
		}
	}

	return mst;
}

// compute tree cotree decomposition.
// enumerate homology basis (2g + (b-1)*(b>=0)).
// warnings : this code does not conceren about disconnected meshes.
// @return cycles (vertex or face list) v_1 -> v_2 -> ... -> v_n
template <bool DualCycle = false>
std::vector<std::vector<uint32_t>> treeCotreeDecomp(
	const heds::HalfedgeMesh& mesh) {
	
	//---------- prepare primal mst ----------

	std::vector<bool> boundaryBridgeMask(mesh.getNumEdges(), true);
	for (const auto& e : mesh.edges) {
		if (!e->isBoundary() && (e->vert->isBoundary() || e->pair->vert->isBoundary()))
			boundaryBridgeMask[e->uid()] = false;
	}

	auto primMst = kruskalSpanningTree(mesh, {}, boundaryBridgeMask);
	std::vector<bool> primMstMask(mesh.getNumEdges(), false);
	for (const auto i : primMst) primMstMask[i] = true;

	//---------- compute cotree ----------

	auto dual = mesh.dualMesh();

	std::vector<bool> cotree(dual.getNumEdges(), false);
	for (const auto& edge : mesh.edges) {
		if (!primMstMask[edge->uid()] && !edge->isBoundary()) {
			cotree[dual.halfedge(edge->adjacentFaces())->uid()] = true;
		}
	}

	//---------- compute dual mst ----------

	auto dualMst = kruskalSpanningTree(dual, {}, cotree);
	std::vector<bool> dualMstMask(dual.getNumEdges(), false);
	for (const auto i : dualMst) dualMstMask[i] = true;

	//---------- find bridges (neither primal mst nor dual mst) ----------

	bool skipFirstBoundary = true;
	std::vector<std::pair<uint32_t, uint32_t>> bridges;
	for (const auto& edge : dual.edges) {
		if (!cotree[edge->uid()] || dualMstMask[edge->uid()]) continue;

		auto&& prim = mesh.halfedge(edge->adjacentFaces());
		if constexpr(DualCycle) {
			if (skipFirstBoundary && !boundaryBridgeMask[prim->uid()]) skipFirstBoundary = false;
			else bridges.emplace_back(edge->endpoint());
		}	else {
			if (!boundaryBridgeMask[prim->uid()]) continue;
			bridges.emplace_back(prim->endpoint());
		}
	}

	if constexpr(!DualCycle) {
		for (const auto& edge : mesh.edges) {
			if (edge->isBoundary() && !primMstMask[edge->uid()]) {
				if (skipFirstBoundary) skipFirstBoundary = false;
				else bridges.emplace_back(edge->endpoint());
			}
		}
	}

	//---------- cycle extraction ----------
	std::vector<std::vector<uint32_t>> adjacentList(mesh.getNumVerts());
	auto constructAdList = [&adjacentList](const auto& hemesh, const auto& mst) {
		adjacentList.resize(hemesh.getNumVerts());
		for (size_t i = 0; i < mst.size(); i++) {
			auto [u, v] = hemesh.edges[mst[i]]->endpoint();
			adjacentList[u].emplace_back(v);
			adjacentList[v].emplace_back(u);
		}
	};
	if constexpr(DualCycle) constructAdList(dual, dualMst);
	else constructAdList(mesh, primMst);
	
	return impl::cyclesExtraction(adjacentList, bridges);
}

// compute minimum homology basis.
// 
// warnings : this code does not concerned about disconnected meshes.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
std::vector<std::vector<uint32_t>> minimumHomologyBasis(
	const heds::SurfaceMesh& mesh, const bool pb = false) {
	//--------------- prepare resource -----------------
	auto dual = mesh.dualMesh();
	
	std::vector<std::vector<uint32_t>> cycles;
	cycles.reserve(mesh.getNumVerts() * 2);

	{
		if (pb) std::cout << "compute homology basis candidates" << std::endl;

		std::priority_queue<std::pair<double, uint32_t>> heap;

		for (uint32_t source = 0; source < mesh.getNumVerts();) {
			//---------- compute shortest path tree ----------
			auto shortestPathTree = kika::gd::DijkstraSolver(mesh).solve({ source });

			std::vector<std::vector<uint32_t>> paths(mesh.getNumVerts());
			std::vector<double> lengths(mesh.getNumVerts());
			std::vector<bool> sptMask(mesh.getNumEdges(), false);

			for (uint32_t i = 0; i < mesh.getNumVerts(); i++) {
				paths[i] = shortestPathTree.root(i);
				lengths[i] = shortestPathTree.value(i);
				if (i == source) continue;
				sptMask[mesh.halfedge(i, paths[i][1])->uid()] = true;
			}

			//---------- enumerate cycle candidates (greedy tree-cotree decomposition) ----------

			std::vector<double> cycleLengths(dual.getNumEdges(), std::numeric_limits<double>::max());
			std::vector<bool> cotree(dual.getNumEdges(), false);

			for (const auto& edge : mesh.edges) {
				if (sptMask[edge->uid()] || edge->isBoundary()) continue;
				auto [u, v] = edge->endpoint();
				auto eid = dual.halfedge(edge->adjacentFaces())->uid();

				cycleLengths[eid] = -(lengths[u] + lengths[v] + mesh.length(edge));
				cotree[eid] = true;
			}


			auto maxSpanningTree = kruskalSpanningTree(dual, cycleLengths, cotree);
			std::vector<bool> mstMask(dual.getNumEdges(), false);
			for (const auto i : maxSpanningTree) mstMask[i] = true;

			std::vector<std::pair<uint32_t, uint32_t>> bridges;
			for (const auto& edge : dual.edges) {
				if (!cotree[edge->uid()] || mstMask[edge->uid()] || edge->isBoundary()) continue;
				bridges.emplace_back(mesh.halfedge(edge->adjacentFaces())->endpoint());
				heap.emplace(cycleLengths[edge->uid()], (uint32_t)heap.size());
			}

			std::vector<std::vector<uint32_t>> adList(mesh.getNumVerts());
			for (size_t i = 0; i < sptMask.size(); i++) {
				if (!sptMask[i]) continue;
				auto [u, v] = mesh.edges[i]->endpoint();
				adList[u].emplace_back(v);
				adList[v].emplace_back(u);
			}

			auto tct = impl::cyclesExtraction(adList, bridges);
			cycles.insert(cycles.end(), tct.begin(), tct.end());

			source++;
			if (pb) util::printProgress((double)source / mesh.getNumVerts());
		}

		std::vector<std::vector<uint32_t>> sortedCycles(cycles.size());
		for (size_t i = 0; i < sortedCycles.size(); i++) {
			sortedCycles[i] = cycles[heap.top().second];
			heap.pop();
		}
		cycles = sortedCycles;
	}
	
	////--------------- test intersection (homology signature) -----------------
	
	auto dualCycles = treeCotreeDecomp<true>(mesh);
	
	std::vector<std::vector<bool>> dualCyclesMask(dualCycles.size(),
		std::vector<bool>(dual.getNumEdges(), false));
	for (size_t i = 0; const auto& cycle : dualCycles) {
		for (size_t j = 0, k = cycle.size(); j < k; j++) {
			uint32_t u = (uint32_t)cycle[j], v = (uint32_t)cycle[(j+1)%k];
			dualCyclesMask[i][dual.halfedge(u, v)->uid()] = true;
		}
		i++;
	}
	
	Eigen::Matrix<bool, -1, -1> E =
		Eigen::Matrix<bool, -1, -1>::Zero(dualCycles.size(), cycles.size());
	
	for (size_t i = 0; i < cycles.size(); i++) {
		for (size_t j = 0, k = cycles[i].size(); j < k; j++) {
			uint32_t u = (uint32_t)cycles[i][j], v = (uint32_t)cycles[i][(j+1)%k];
	
			auto ei = dual.halfedge(mesh.halfedge(u, v)->adjacentFaces())->uid();
			for (size_t g = 0; g < dualCycles.size(); g++) {
				if (dualCyclesMask[g][ei]) E(g, i) = !E(g, i);
			}
		}
	}
	
	impl::gaussJordanElimZ2(E, pb);
	
	std::vector<std::vector<uint32_t>> basis;
	for (size_t i = 0, j = 0; (i < cycles.size()) && (j < dualCycles.size()); i++) {
		if (E(j, i)) {
			basis.emplace_back(cycles[i]);
			j++;
		}
	}
	
	return basis;
}

template <typename Mesh = heds::HalfedgeMesh>
class HomologyGenerator {
	Mesh& mesh;

public:
	HomologyGenerator(const Mesh& mesh) : mesh(mesh) {}

	template <bool DualCycle = false>
	std::vector<std::vector<uint32_t>> treeCotreeDecomp() const {

	}

	std::vector<std::vector<uint32_t>> minimumBasis() const {

	}

};

}

namespace kika::topo {

enum class MinCycleOpt { All, NonSeparating, NonContractible, 
	SeparatingAndNonContractible, NonSeparatingAndContractible };
using enum MinCycleOpt;

namespace impl {

template <MinCycleOpt Opt>
struct CycleCondition {
	// check the number of genus on separated mesh
	int countGenus(const heds::HalfedgeMesh& mesh, const std::vector<bool>& mask) const {
		int v = 0, e = 0, f = 0, b = 1; // (b = 1, at least one boundary generated by cycle)

		for (size_t i = 0; i < mask.size(); i++) v += mask[i];

		for (const auto& face : mesh.faces) {
			bool flag = true;
			for (const auto& vert : face->circuitVerts()) {
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

	// get interior and exterior vertex of cycle.
	// @return condition 0 : normal end.
	//                   1 : cycle hasn't interior vertex and non-boundary.
	//                   2 : cycle is boundary.
	int getInteriorAndExteriorOfCycle(std::vector<uint32_t>& queue, const heds::HalfedgeMesh& mesh, 
		const std::vector<uint32_t>& cycle, const std::vector<bool>& cycleVerts) const {

		bool leftSide = true, rightSide = true;
		bool leftSideIsBoundary = true, rightSideIsBoundary = true;
		for (size_t i = 0, j = cycle.size(); i < j; i++) {
			auto u = cycle[i], v = cycle[(i+1)%j];
			auto edge = mesh.halfedge(u, v);
			leftSideIsBoundary &= edge->face == nullptr;
			if (leftSide && edge->face != nullptr && !cycleVerts[edge->next->vert->id]) {
				queue.emplace_back(edge->next->vert->id);
				leftSide = false;
			}
			rightSideIsBoundary &= edge->pair->face == nullptr;
			if (rightSide && edge->pair->face != nullptr && !cycleVerts[edge->pair->next->vert->id]) {
				queue.emplace_back(edge->pair->next->vert->id);
				rightSide = false;
			}
			if (!leftSide && !rightSide) break;
		}
		if (queue.size() < 2) {
			return (leftSideIsBoundary || rightSideIsBoundary) + 1;
		}
		return 0;
	}

	// check left and right side surfaces doesn't separeted.
	bool isCycleSeparating(std::vector<uint32_t>&& queue, const heds::HalfedgeMesh& mesh, 
		const std::vector<bool>& cycleVerts, std::vector<uint32_t>& surfaceGroup) const {

		surfaceGroup.resize(mesh.getNumVerts(), 0);
		for (uint32_t i = 0; i < queue.size(); i++) surfaceGroup[queue[i]] = i + 1;

		while (!queue.empty()) {
			auto i = queue.back();
			queue.pop_back();

			for (const auto& v : mesh.verts[i]->oneringNeighbs()) {
				if (surfaceGroup[v] != 0 && (surfaceGroup[v] != surfaceGroup[i])) return false;
				
				if (surfaceGroup[v] == 0 && !cycleVerts[v]) {
					surfaceGroup[v] = surfaceGroup[i];
					queue.emplace_back(v);
				}
			}
		}

		return true;
	}

	// surfaces are non-zero genus or has boundary
	bool isSurfaceContractible(const heds::HalfedgeMesh& mesh, 
		const std::vector<uint32_t>& surfaceGroup, const int group) const {
		std::vector<bool> mask(mesh.getNumVerts(), false);
		for (size_t i = 0; i < mesh.getNumVerts(); i++) {
			mask[i] = (surfaceGroup[i] == group);
			if (mask[i] && mesh.verts[i]->isBoundary()) return false;
		}
		return countGenus(mesh, mask) == 0;
	}

	// (non-separating test) check left-side and right-side surfaces of cycle doesn't separeted.
	// (non-contractible test) separated surfaces by cycle are non-zero genus or has boundary.
	// return (1)     : cycle hasn't interior vertex and is non-boundary -> (separating and contractible).
	// return (2)(ns) : cycle is boundary -> (non-separating).
	// return (3)(snc): cycle is boundary -> (non-separating).
	// return (4)(nsc): cycle is boundary and mesh has one boundary and zero genus -> (non-separating and contractible).
	// return (5)(ns) : both surfaces aren't separeted -> (non-separating).
	// return (6)(nc) : both surfaces aren't separeted -> (non-contractible).
	// return (7)(snc): both surfaces aren't separeted -> (non-separating).
	// return (8)(nc) : both surfaces are non-zero genus or has boundary -> (non-contractible).
	bool satisfy(const heds::HalfedgeMesh& mesh, const std::vector<uint32_t>& cycle) const {
		std::vector<bool> cycleVerts(mesh.getNumVerts(), false);
		for (auto& i : cycle) cycleVerts[i] = true;

		std::vector<uint32_t> queue;
		constexpr int noInterior = 1, isBoundary = 2;
		auto cycleStatus = getInteriorAndExteriorOfCycle(queue, mesh, cycle, cycleVerts);

		if (cycleStatus == noInterior) return false; // (1)
		if constexpr (Opt == NonSeparating) if (cycleStatus == isBoundary) return true; // (2)
		if constexpr (Opt == SeparatingAndNonContractible) if (cycleStatus == isBoundary) return false; // (3)
		if constexpr (Opt == NonSeparatingAndContractible) 
			return (cycleStatus == isBoundary) && (mesh.getNumBoundaryCycles() == 1) && (mesh.genus() == 0); // (4)

		std::vector<uint32_t> surfaceGroup;
		auto isSeparating = isCycleSeparating(std::move(queue), mesh, cycleVerts, surfaceGroup);

		if constexpr (Opt == NonSeparating) return !isSeparating; // (5)
		if constexpr (Opt == NonContractible) if (!isSeparating) return true; // (6)
		if constexpr (Opt == SeparatingAndNonContractible) if (!isSeparating) return false; // (7)

		return !isSurfaceContractible(mesh, surfaceGroup, 1) && ((cycleStatus == isBoundary) || 
			     !isSurfaceContractible(mesh, surfaceGroup, 2)); // (8)
	}

public:
	bool value;

	CycleCondition(const heds::HalfedgeMesh& mesh, const std::vector<uint32_t>& cycle) : value(true) {
		if constexpr (Opt == All) {
			value = true;
		} else {
			value = satisfy(mesh, cycle);
		}
	}
};

}

// compute minimum cycle candidates (opt : all, non-separating, non-contractible).
// 
// warnings : this code does not conceren about disconnected meshes.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
template <MinCycleOpt Opt = MinCycleOpt::All>
std::vector<std::vector<uint32_t>> minimumCycleCandidates(
	const heds::SurfaceMesh& mesh, const bool pb = false) {

	std::vector<std::vector<uint32_t>> cycles;
	cycles.reserve(mesh.getNumEdges());

	if (pb) std::cout << "compute candidates" << std::endl;

	std::vector<std::vector<uint32_t>> paths(mesh.getNumVerts());
	std::priority_queue<std::pair<double, uint32_t>> heap;

	for (uint32_t source = 0; source < mesh.getNumVerts();) {
		auto spt = kika::gd::DijkstraSolver(mesh).solve({ source });
		
		for (uint32_t i = 0; i < paths.size(); i++)
			paths[i] = spt.root(i);

		for (const auto& edge : mesh.edges) {
			auto [u, v] = edge->endpoint();

			if (u == source || v == source) continue;

			auto path1 = paths[u];
			auto path2 = paths[v];

			// check cycle is horton's cycle 
			//(path2 is equal to (odd) or +- one (even) length of path1.)
			if (abs((int)path1.size() - (int)path2.size()) >= 2) continue;

			// check cycle is internally disjoint
			if (*(path1.rbegin() + 1) == *(path2.rbegin() + 1)) continue;

			std::vector<uint32_t> cycle(path1.size() + path2.size() - 1);
			std::copy(path2.begin(), path2.end(), cycle.begin());
			std::copy(path1.rbegin() + 1, path1.rend(), cycle.begin() + path2.size());

			// check cycle unsatisfy condition
			if (impl::CycleCondition<Opt>(mesh, cycle).value) {
				cycles.emplace_back(std::move(cycle));
				heap.emplace(-(spt.value(u) + spt.value(v) + mesh.length(edge)), (uint32_t)cycles.size() - 1);
			}
		}

		source++;
		if (pb) util::printProgress((double)source / mesh.getNumVerts());
	}
	if (pb) std::cout << std::endl;

	std::vector<std::vector<uint32_t>> sortedCycles(cycles.size());
	for (size_t i = 0; i < sortedCycles.size(); i++) {
		sortedCycles[i] = cycles[heap.top().second];
		heap.pop();
	}
	return sortedCycles;
}

// compute minimum cycle basis.
// 
// warnings : this code does not conceren about disconnected meshes.
// @return cycles (vertex list) v_1 -> v_2 -> ... -> v_n
template <MinCycleOpt Opt = MinCycleOpt::All>
std::vector<std::vector<uint32_t>> minimumCycleBasis(
	const heds::SurfaceMesh& mesh, const bool pb = false) {
	auto cycles = minimumCycleCandidates<Opt>(mesh, pb);

	Eigen::MatrixX<bool> E =	Eigen::MatrixX<bool>::Zero(mesh.getNumEdges(), cycles.size());
	for (size_t i = 0; i < cycles.size(); i++) {
		for (size_t j = 0, k = cycles[i].size(); j < k; j++) {
			uint32_t u = (uint32_t)cycles[i][j], v = (uint32_t)cycles[i][(j+1)%k];
			E(mesh.halfedge(u, v)->uid(), i) = true;
		}
	}

	impl::gaussJordanElimZ2(E, pb);

	std::vector<std::vector<uint32_t>> basis;
	for (size_t i = 0, j = 0; (i < cycles.size()) && (j < mesh.getNumEdges()); i++) {
		if (E(j, i)) {
			basis.emplace_back(cycles[i]);
			j++;
		}
	}

	return basis;
}

}