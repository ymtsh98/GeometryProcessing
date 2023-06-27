#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <kika/az_memory_pool.hpp>

/// type definition
namespace kika::heds {
class Vertex;
class Halfedge;
class Face;

using ptv = Vertex*;
using pte = Halfedge*;
using ptf = Face*;

using ptVerts = std::vector<ptv>;
using ptEdges = std::vector<pte>;
using ptFaces = std::vector<ptf>;

enum class ReturnType {
	Idx,
	Ptr
};

enum class ReturnOrder {
	CW,
	CCW
};

using enum ReturnType;
using enum ReturnOrder;

using EdgeKey = std::pair<uint32_t, uint32_t>;
inline EdgeKey keyPair(const EdgeKey& key) { return EdgeKey(key.second, key.first); }
struct EdgeKeyHash {
	uint64_t operator()(const EdgeKey& p) const {
		return static_cast<uint64_t>(p.first) << 32 | p.second;
	}
};
using EdgeKeyMap  = std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash>;
//using EdgeKeyMap = absl::flat_hash_map<EdgeKey, uint32_t>;
//using EdgeKeyMap = robin_hood::unordered_map<EdgeKey, uint32_t, EdgeKeyHash>;

////using EdgeKey = uint64_t;
////inline EdgeKey makeEdgeKey(const uint32_t& u, const uint32_t& v) { return (static_cast<uint64_t>(u) << 32) | v; }
////inline EdgeKey keyPair(const EdgeKey& key) { return (key << 32) | (key >> 32); }

}

/// elements
namespace kika::heds {
class Vertex {
public:
	pte edge;
	uint32_t id;
	
	Vertex() : edge(nullptr), id(0) {}
	Vertex(const uint32_t id) : edge(nullptr), id(id) {}
	~Vertex() { edge = nullptr; }

	// get one-ring neighbors (default : ordered clockwise)
	template <ReturnType Type = Idx, ReturnOrder Order = CW, 
		typename Ret = std::conditional_t<Type == heds::Idx, std::vector<uint32_t>, ptVerts>>
	Ret	oneringNeighbs() const;

	// get incident edges (default : ordered clockwise)
	template <ReturnType Type = Idx, ReturnOrder Order = CW, 
		typename Ret = std::conditional_t<Type == heds::Idx, std::vector<uint32_t>, ptEdges>>
	Ret	incidentEdges() const;

	// get adjacent faces (default : ordered clockwise)
	template <ReturnType Type = Idx, ReturnOrder Order = CW, 
		typename Ret = std::conditional_t<Type == heds::Idx, std::vector<uint32_t>, ptFaces>>
	Ret	adjacentFaces() const;

	// get num of onerings
	size_t degree() const;
	
	bool isBoundary() const;

	// is unreferenced vertex
	bool isUnrefer() const;

	// all adjacent faces are removable
	bool isRemovable() const;
};

class Halfedge {
public:
	pte next;
	pte pair;
	ptv vert;
	ptf face;
	uint32_t id;

	Halfedge() : next(nullptr), pair(nullptr), vert(nullptr), face(nullptr), id(0) {}
	Halfedge(const uint32_t id) : next(nullptr), pair(nullptr), vert(nullptr), face(nullptr), id(id) {}
	~Halfedge() { next = nullptr; pair = nullptr; vert = nullptr; face = nullptr; }

	// previous edge
	pte prev() const;

	// unique index (id / 2)
	uint32_t uid() const;

	// is secondaly edge (id % 2), (if true : this edge is not in mesh.edges)
	bool isSecond() const;

	// get adjacent faces (left, right)
	template <ReturnType Type = Idx, 
		typename Ret = std::conditional_t<Type == Idx, std::pair<uint32_t, uint32_t>, std::pair<ptf, ptf>>>
	Ret adjacentFaces() const;

	// get endpoint vertex (source, target)
	template <ReturnType Type = Idx, 
		typename Ret = std::conditional_t<Type == Idx, std::pair<uint32_t, uint32_t>, std::pair<ptv, ptv>>>
	Ret endpoint() const;

	// check this != pair->pair 
	bool isInvalid() const;

	bool isBoundary() const;

	// unremovable : adjacent faces has more than 2 sharing edges.
	bool isRemovable() const;

	// check this is part of K4 (inner / outer edge).
	template <bool InnerTest = true, bool OuterTest = true>
	bool isK4() const;

	// check this edge part of tetrahedral
	bool isTet() const;

	// edge flip
	void flip();

};

class Face {
public:
	pte edge;
	uint32_t id;

	Face() : edge(nullptr), id(0) {}
	Face(const uint32_t id) : edge(nullptr), id(id) {}
	~Face() { edge = nullptr; }

	// get circuit vertices (default : ordered counter-clockwise)
	template <ReturnType Type = Idx, ReturnOrder Order = CCW,
		typename Ret = std::conditional_t<Type == heds::Idx, std::vector<uint32_t>, ptVerts>>
	Ret circuitVerts() const;

	// get circuit edges (default : ordered counter-clockwise)
	template <ReturnType Type = Idx, ReturnOrder Order = CCW,
		typename Ret = std::conditional_t<Type == heds::Idx, std::vector<uint32_t>, ptEdges>>
	Ret circuitEdges() const;

	// get adjacent faces (default : ordered counter-clockwise)
	template <ReturnType Type = Idx, ReturnOrder Order = CCW,
		typename Ret = std::conditional_t<Type == heds::Idx, std::vector<uint32_t>, ptFaces>>
	Ret adjacentFaces() const;

	// get num of vertices
	size_t ngon() const;

	// check this face is invalid (all edges are valid)
	bool isInvalid() const;
	
	bool isBoundary() const;

	// unremovable : any vertex is boundary, and its side edges are non boundary.
	bool isRemovable() const;
};


}

/// impl elements
namespace kika::heds {

namespace impl {
template <ReturnType Type, typename V, typename E = V::value_type>
inline constexpr void appendIdxOrPtr(V& v, E& e) {
	if constexpr(Type == heds::Idx) v.emplace_back(e->id);
	else v.emplace_back(e);
}
}

// -------------------- Vertex --------------------

// get one-ring neighbors (default : ordered clockwise)
template <ReturnType Type, ReturnOrder Order, typename Ret>
Ret Vertex::oneringNeighbs() const {
	Ret neighbs;
	neighbs.reserve(6);
	pte e = this->edge;

	do {
		impl::appendIdxOrPtr<Type>(neighbs, e->pair->vert);
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) impl::appendIdxOrPtr<Type>(neighbs, e->pair->vert);

	if constexpr (Order == CCW) std::reverse(neighbs.begin() + 1, neighbs.end());
	return neighbs;
}

// get incident edges (default : ordered clockwise)
template <ReturnType Type, ReturnOrder Order, typename Ret>
Ret Vertex::incidentEdges() const {
	Ret edges;
	edges.reserve(6);
	pte e = this->edge;
	
	do {
		impl::appendIdxOrPtr<Type>(edges, e);
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) impl::appendIdxOrPtr<Type>(edges, e);

	if constexpr (Order == CCW) std::reverse(edges.begin() + 1, edges.end());
	return edges;
}

// get adjacent faces (default : ordered clockwise)
template <ReturnType Type, ReturnOrder Order, typename Ret>
Ret Vertex::adjacentFaces() const {
	Ret faces;
	faces.reserve(6);
	pte e = this->edge;

	do {
		impl::appendIdxOrPtr<Type>(faces, e->face);
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);

	if constexpr (Order == CCW) std::reverse(faces.begin() + 1, faces.end());
	return faces;
}

// get num of onerings
inline size_t Vertex::degree() const {
	size_t d = 0;
	pte e = this->edge;

	do {
		d++;
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) d++;

	return d;
}

inline bool Vertex::isBoundary() const {
	return edge->pair->face == nullptr;
}

// is unreferenced vertex
inline bool Vertex::isUnrefer() const {
	return edge == nullptr || edge->face == nullptr;
}

// (check onering face is removable.)
inline bool Vertex::isRemovable() const {
	for (auto& e : incidentEdges<Ptr>()) {
		if (!e->isBoundary() && e->pair->vert->isBoundary() &&
			!e->pair->next->isBoundary() && !e->prev()->isBoundary()) return false;
	}
	return true;
}

// -------------------- Halfedge --------------------

// previous edge
inline pte Halfedge::prev() const {
	pte e = this->next;
	while (e->next != this) e = e->next;
	return e;
}

// unique index (id / 2)
inline uint32_t Halfedge::uid() const {
	return id / 2;
}

// is prim edge (id % 2)
inline bool Halfedge::isSecond() const {
	return id % 2;
}

// get adjacent faces (left, right)
template <ReturnType Type, typename Ret>
Ret Halfedge::adjacentFaces() const {
	if constexpr (Type == Idx) 
		return Ret{ this->face == nullptr ? std::numeric_limits<uint32_t>::max() : this->face->id, 
		            this->pair->face == nullptr ? std::numeric_limits<uint32_t>::max() : this->pair->face->id };
	else 
		return Ret{ this->face, this->pair->face };
}

// get endpoint vertex (source, target)
template <ReturnType Type, typename Ret>
Ret Halfedge::endpoint() const {
	if constexpr(Type == Idx) return Ret{ this->pair->vert->id, this->vert->id };
	else return Ret{ this->pair->vert, this->vert };
}

// check this edge != pair->pair 
inline bool Halfedge::isInvalid() const {
	return this != this->pair->pair;
}

inline bool Halfedge::isBoundary() const {
	return this->face == nullptr || this->pair->face == nullptr;
}

// unremovable : adjacent faces has more than 2 sharing edges.
inline bool Halfedge::isRemovable() const {
	auto rf = pair->face;
	if (rf == nullptr) return true;

	auto l = this->face->circuitVerts();
	std::sort(l.begin(), l.end());
	auto r = rf->circuitVerts();
	std::sort(r.begin(), r.end());
	kika::util::pushBackCounter<uint32_t> pbc;
	std::set_intersection(l.begin(), l.end(), r.begin(), r.end(), std::back_inserter(pbc));
	return pbc.count <= 2;
}

#if false
// triangle only
inline void Halfedge::contractSwitch() {
	auto e1 = this->next;
	auto e2 = e1->next;
	if (e1->vert->edge == e1) e1->vert->edge = e2->pair;
	e1 = e1->pair;
	e2 = e2->pair;

	e1->pair = e2;
	e2->pair = e1;
}

// triangle only
inline void Halfedge::contract(ptv& vert) {
	auto&& es0 = this->vert->incidentEdges();
	auto&& es1 = this->pair->vert->incidentEdges();

	if (this->face != nullptr) this->contractSwitch();
	if (this->pair->face != nullptr) this->pair->contractSwitch();

	for (const auto& e : es0) e->vert = vert;
	for (const auto& e : es1) e->vert = vert;

	if (es0[0]->pair->face == nullptr && es1[0]->pair->face == nullptr) {
		vert->edge = (es0[0] == this->pair || es0[0]->pair == this->pair) ? es1[0] : es0[0];
	} else {
		vert->edge = es0[0]->pair->face == nullptr ? es0[0] : es1[0];
		if (vert->edge == this->pair || vert->edge->pair == this->pair) vert->edge = vert->edge->next->pair;
	}
	vert->edge = vert->edge->pair->pair;
}

// triangle only
inline bool Halfedge::isContractible() const {
	int count = isBoundary();

	if (count == 0 && vert->isBoundary() && pair->vert->isBoundary()) return false;

	auto&& oring = vert->oneringNeighbs();
	std::unordered_set<ptv> set;
	set.reserve(oring.size());
	for (const auto& v : oring) set.emplace(v);
	for (const auto& v : pair->vert->oneringNeighbs()) count += set.contains(v);
	return count < 3;
}
#endif

template <bool InnerTest, bool OuterTest>
bool Halfedge::isK4() const {
	auto&& ov = vert->oneringNeighbs(), ou = pair->vert->oneringNeighbs();
	if constexpr (InnerTest) {
		if (ov.size() == 3 || ou.size() == 3) return true;
	} 
	if constexpr (OuterTest) {
		// check each end point shares over 2 vertices.
		std::sort(ov.begin(), ov.end());
		std::sort(ou.begin(), ou.end());
		kika::util::pushBackCounter<uint32_t> pbc;
		std::set_intersection(ov.begin(), ov.end(), ou.begin(), ou.end(), std::back_inserter(pbc));
		return pbc.count > 2;
	}
	return false;
}

// check this edge part of tetrahedral
inline bool Halfedge::isTet() const {
	auto&& ov = next->vert->oneringNeighbs();
	auto& u = pair->next->vert->id;
	for (const auto& v : ov) {
		if (v == u) return true;
	}
	return false;
}

// edge flip
inline void Halfedge::flip() {
	if (isBoundary()) return;
	pte a = next, b = prev(), c = pair->next, d = pair->prev();

	// vert -> edge
	if (vert->edge == this) vert->edge = d;
	if (pair->vert->edge == pair) pair->vert->edge = b;

	// this -> vert
	vert = a->vert;
	pair->vert = c->vert;

	// face -> edge
	face->edge = this;
	pair->face->edge = pair;

	// prev -> next
	b->next = c;
	d->next = a;

	// next -> face
	a->face = pair->face;
	c->face = face;

	// this -> next
	next = a->next;
	pair->next = c->next;

	// next -> next
	a->next = pair;
	c->next = this;
}

// -------------------- Face --------------------

// get circuit vertices (default : ordered counter-clockwise)
template <ReturnType Type, ReturnOrder Order, typename Ret>
Ret Face::circuitVerts() const {
	Ret verts;
	verts.reserve(3);
	pte e = this->edge;

	do {
		impl::appendIdxOrPtr<Type>(verts, e->vert);
		e = e->next;
	} while (e != this->edge);

	if constexpr (Order == CW) std::reverse(verts.begin() + 1, verts.end());
	return verts;
}

// get circuit edges (default : ordered counter-clockwise)
template <ReturnType Type, ReturnOrder Order, typename Ret>
Ret Face::circuitEdges() const {
	Ret edges;
	edges.reserve(3);
	pte e = this->edge;

	do {
		impl::appendIdxOrPtr<Type>(edges, e);
		e = e->next;
	} while (e != this->edge);

	if constexpr (Order == CW) std::reverse(edges.begin() + 1, edges.end());
	return edges;
}

// get adjacent faces (default : ordered counter-clockwise)
template <ReturnType Type, ReturnOrder Order, typename Ret>
Ret Face::adjacentFaces() const {
	Ret faces;
	faces.reserve(3);
	pte e = this->edge;

	do {
		if (e->pair->face != nullptr) 
			impl::appendIdxOrPtr<Type>(faces, e->pair->face);
		e = e->next;
	} while (e != this->edge);

	if constexpr (Order == CW) std::reverse(faces.begin() + 1, faces.end());
	return faces;
}

// get num of vertices
inline size_t Face::ngon() const {
	size_t n = 0;
	pte e = this->edge;
	do {
		n++;
		e = e->next;
	} while (e != this->edge);
	return n;
}

// check this face is invalid (any edge is invalid)
inline bool Face::isInvalid() const {
	pte e = this->edge;
	do {
		if (e->isInvalid()) return true;
		e = e->next;
	} while (e != this->edge);
	return false;
}

inline bool Face::isBoundary() const {
	pte e = this->edge;
	do {
		if (e->pair->face == nullptr) return true;
		e = e->next;
	} while (e != this->edge);
	return false;
}

// unremovable : any vertex is boundary, and its side edges are non boundary.
inline bool Face::isRemovable() const {
	for (const auto& e : this->circuitEdges<Ptr>()) {
		if ((!e->isBoundary() && !e->next->isBoundary()) && e->vert->isBoundary()) return false;
	}

	return true;
}

}

/// halfedge mesh
namespace kika::heds {

/// halfedge mesh class
class HalfedgeMesh {
protected:
	util::AZMemoryPool<Vertex, Halfedge, Face> mpool;
public:
	ptVerts verts;
	ptEdges edges;
	ptFaces faces;

private:
	template <typename T>
	auto make(const uint32_t i) {
		return new(mpool.bind<T>()) T(i);
	}

	template <typename T>
	inline size_t faceInputSize(const T& data) const {
		if constexpr (util::IsEigen<T>) return static_cast<size_t>(data.rows());
		else if constexpr (util::IsStdvec2d<T>) return static_cast<size_t>(data.size());
		else return static_cast<size_t>(data.size() / 3);
	}

	template <typename T>
	void initVertices(const T& nv) {
		verts.resize(nv);

		for (uint32_t i = 0; i < verts.size(); i++) {
			verts[i] = make<Vertex>(i);
		}
	}

	template <typename T>
	void initFaces(const T& faceAttrib) {
		uint32_t n = static_cast<uint32_t>(faceInputSize(faceAttrib)), edgeIndex = 0;

		faces.reserve(n);
		edges.reserve(3*n/2);

		EdgeKeyMap keymap;
		keymap.reserve(3 * n / 2);

		std::vector<EdgeKey> feKey; feKey.reserve(3);
		for (uint32_t i = 0; i < n; i++) {
			feKey.clear();
			if constexpr (util::IsEigen<T>) {
				for (size_t j = 0, k = (size_t)faceAttrib.cols(); j < k; j++)
					feKey.emplace_back(faceAttrib(i, j), faceAttrib(i, (j+1)%k));
			} else if constexpr (util::IsStdvec2d<T>) {
				for (size_t j = 0, k = faceAttrib[i].size(); j < k; j++)
					feKey.emplace_back(faceAttrib[i][j], faceAttrib[i][(j+1)%k]);
			} else {
				feKey.emplace_back(faceAttrib[i * 3 + 0], faceAttrib[i * 3 + 1]);
				feKey.emplace_back(faceAttrib[i * 3 + 1], faceAttrib[i * 3 + 2]);
				feKey.emplace_back(faceAttrib[i * 3 + 2], faceAttrib[i * 3 + 0]);
			}

			faces.emplace_back(make<Face>(i));

			if (feKey.size() == 0) continue;

			pte front = nullptr, prev = nullptr;
			for (size_t j = 0; j < feKey.size(); j++) {
				auto& first = verts[feKey[j].first], & second = verts[feKey[j].second];

				pte edge;
				auto find = keymap.find(keyPair(feKey[j]));
				if (find != keymap.end()) {
					edge = edges[find->second]->pair;
				} else {
					keymap.emplace(feKey[j], edgeIndex/2);
					edge = edges.emplace_back(make<Halfedge>(edgeIndex++));
					edge->pair = make<Halfedge>(edgeIndex++);
					edge->pair->pair = edge;
					edge->pair->vert = first;
					edge->vert = second;
				}

				if (j == 0) front = edge;
				else prev->next = edge;
				prev = edge;

				edge->face = faces[i];
				faces[i]->edge = edge;
				second->edge = edge;
			}

			if (front != nullptr) prev->next = front;
		}
	}

	void initBoundary() {
		for (const auto& e : edges) {
			if (e->pair->face == nullptr)
				verts[e->vert->id]->edge = e;
		}
	}

public:

	/// halfedge ds mesh
	HalfedgeMesh() {}

	/// halfedge ds mesh
	// @param nv	: number of vertices (scalar)
	// @param faceAttrib	: faces list (2d) or index list (1d)
	template<typename T1, typename T2, class = std::enable_if_t<std::is_scalar_v<T1>, void>>
	HalfedgeMesh(const T1& nv, const T2& faceAttrib) {
		build(nv, faceAttrib);
	}

	HalfedgeMesh(const HalfedgeMesh& mesh) {
		build(mesh.getNumVerts(), mesh.faceList());
	}

	HalfedgeMesh& operator=(const HalfedgeMesh& mesh) {
		reset(true);
		build(mesh.getNumVerts(), mesh.faceList());
		return *this;
	}

	~HalfedgeMesh() {
		reset(false);
	}

	// builds halfedge ds mesh
	// @param nv	: number of vertices (scalar)
	// @param faceAttrib	: face matrix (2d) or index list (1d)
	template<typename T1, typename T2, class = std::enable_if_t<std::is_scalar_v<T1>, void>>
	void build(const T1& nv, const T2& faceAttrib) {
		size_t nf = faceInputSize(faceAttrib), ne = 3 * nf;
		mpool.ialloc((size_t)nv, ne, nf);

		initVertices((size_t)nv);
		initFaces(faceAttrib);
		initBoundary();
	}

	void reset(bool resizeObject = true) {
		//// (if ptv, pte, ptf are shared_ptr)
		//for (auto& v : verts) v->~Vertex();
		//for (auto& e : edges) {
		//	e->pair->~Halfedge(); e->~Halfedge();
		//}
		//for (auto& f : faces) f->~Face();

		if (resizeObject) {
			verts = ptVerts{};
			faces = ptFaces{};
			edges = ptEdges{};
		}

		mpool.reset();
	}

	// ----------------- mesh information ---------------------

	size_t getNumVerts() const {
		return verts.size();
	}

	size_t getNumEdges() const {
		return edges.size();
	}

	size_t getNumFaces() const {
		return faces.size();
	}

	std::tuple<size_t, size_t, size_t> vef() const {
		return std::make_tuple(getNumVerts(), getNumEdges(), getNumFaces());
	}

	// v - e + f = x
	int eulerPoincare() const {
		return (int)getNumVerts() - (int)getNumEdges() + (int)getNumFaces();
	}
	
	// -(v - e + f  - 2 + b) / 2 = g
	int genus() const {
		return -(eulerPoincare() - 2 + (int)getNumBoundaryCycles()) / 2;
	}

	// get halfedge pointer (if there is no halfedge, return nullptr)
	pte halfedge(const uint32_t s, const uint32_t t) const {
		if (t >= getNumVerts()) return nullptr;
		pte edge = nullptr;
		for (const auto& e : verts[t]->incidentEdges<Ptr>()) {
			if (e->pair->vert->id == s) {
				edge = e;
				break;
			}
		}

		return edge;
	}

	// get halfedge pointer (if there is no halfedge, return nullptr)
	template <typename T1, typename T2>
	pte halfedge(const std::pair<T1, T2>& e) const {
		return halfedge((uint32_t)e.first, (uint32_t)e.second);
	}

	// vertex adjacent list
	template <typename T = uint32_t>
	std::vector<std::vector<T>> adjacencyList(const bool self = false) const {
		std::vector<std::vector<T>> adlist(this->getNumVerts());

		for (size_t i = 0; i < getNumVerts(); i++) {
			for (const auto& v: verts[i]->oneringNeighbs()) adlist[i].emplace_back(static_cast<T>(v));
			if (self) adlist[i].emplace_back(static_cast<T>(i));
		}
		return adlist;
	}

	// face index list
	template <typename T=uint32_t>
	std::vector<std::vector<T>> faceList() const {
		std::vector<std::vector<T>>	flist(getNumFaces());

		for (size_t i = 0; i < getNumFaces(); i++) {
			for (const auto& v : faces[i]->circuitVerts())
				flist[i].emplace_back(static_cast<T>(v));
		}
		return flist;
	}

	// k ring neighbors
	template <typename T = uint32_t>
	std::vector<T> kRingNeighbs(const size_t src, const size_t k) const {
		std::vector<T> neighbs;

		std::deque<std::pair<T, T>> search;
		search.emplace_back(0, src);
		std::vector<bool> arrived(getNumVerts(), false);
		arrived[src] = true;

		while (!search.empty()) {
			auto [d, v] = search.front();
			search.pop_front();

			if (d == k) continue;
			for (const auto& u : verts[v]->oneringNeighbs()) {
				if (!arrived[u]) {
					search.emplace_back(d + 1, u);
					neighbs.emplace_back(u);
					arrived[u] = true;
				}
			}
		}

		return neighbs;
	}

	// get boundary cycles (counter-clockwise)
	template <typename T=uint32_t>
	std::vector<std::vector<T>> boundaryCycles() const {
		std::vector<std::vector<T>> boundaries;
		std::vector<bool> checkedEdges(getNumEdges(), false);

		for (const auto& edge : edges) {
			if (!edge->isBoundary() || checkedEdges[edge->uid()]) continue;

			boundaries.push_back({});

			pte e0 = edge, e = e0;
			do {
				boundaries.back().emplace_back((T)e->vert->id);
				checkedEdges[e->uid()] = true;

				e = e->next;
				while (!e->isBoundary()) {
					e = e->pair->next;
				};
			} while (e != e0);

			std::reverse(boundaries.back().begin(), boundaries.back().end());
		}

		return boundaries;
	}

	size_t getNumBoundaryCycles() const {
		return boundaryCycles().size();
	}

	size_t getNumBoundaryEdges() const {
		size_t n = 0;
		for (size_t i = 0; i < getNumEdges(); i++) if (edges[i]->isBoundary()) n++;
		return n;
	}

	// ----------------- element operation ---------------------
	
private:
	template <typename T>
	struct Swapper {
		void operator()(const uint32_t i, std::vector<T>& vec) {
			vec[i] = vec.back();
			vec.pop_back();
		}
	};

	// remove element by swap and pop_back.
	template <typename T, typename SwapObj = std::vector<uint32_t>, typename SwapFunc = Swapper<typename SwapObj::value_type>>
	void removeElement(T& elem, SwapObj* swapOrder = nullptr) {
		if constexpr (std::is_same_v<T, ptv>) {
			if (swapOrder != nullptr) {
				if constexpr (!std::is_same_v<SwapFunc, void>) SwapFunc{}(elem->id, *swapOrder);
			}
			verts.back()->id = elem->id;
			verts[elem->id] = verts.back();
			verts.pop_back();
			elem->~Vertex();
		} else if constexpr (std::is_same_v<T, pte>) {
			auto id = elem->uid();
			edges.back()->id = 2 * id;
			edges.back()->pair->id = 2 * id + 1;
			edges[id] = edges.back();
			edges.pop_back();
			elem->pair->~Halfedge();
			elem->~Halfedge();
		} else {
			faces.back()->id = elem->id;
			faces[elem->id] = faces.back();
			faces.pop_back();
			elem->~Face();
		}
	}

	//// test remove operators
	// 
	//std::cout << mesh.getNumVerts() << std::endl;
	// 
	// std::random_device seed;
	// std::mt19937 engine(seed());
	// 
	//size_t n = (size_t)std::stoi(argv[2]);
	//auto swapIndices = kika::util::vecrange(mesh.getNumVerts());
	//
	//for (size_t i = 0; i < n; i++) {
	//	if (mesh.getNumVerts() == 0) break;
	//	std::uniform_int_distribution<size_t> distrib(0ull, mesh.getNumVerts() - 1);
	//	mesh.remove(mesh.verts[distrib(engine)], &swapIndices);
	//
	//	if (mesh.getNumFaces() == 0) break;
	//	std::uniform_int_distribution<size_t> distrib(0ull, mesh.getNumFaces() - 1);
	//	mesh.remove(mesh.faces[distrib(engine)], &swapIndices);
	//  
	//	if (mesh.getNumEdges() == 0) break;
	//	std::uniform_int_distribution<size_t> distrib(0ull, mesh.getNumEdges() - 1);
	//	mesh.remove(mesh.edges[distrib(engine)], &swapIndices);
	//}
	//
	//std::cout << mesh.getNumVerts() << std::endl;
	//
	//for (size_t i = 0; i < mesh.getNumVerts(); i++) {
	//  if (i != mesh.verts[i]->id) std::cout << "v : " << i  << std::endl;
	//  if (mesh.verts[i]->isUnrefer()) std::cout << "vinv : " << i << std::endl;
	//}
	//for (size_t i = 0; i < mesh.getNumEdges(); i++) {
	//  if (i*2 != mesh.edges[i]->id || (i*2+1) != mesh.edges[i]->pair->id) std::cout << "e : " << i * 2 << " " << mesh.edges[i]->id << " " << mesh.edges[i]->pair->id  << std::endl;
	//  if (mesh.edges[i]->isInvalid()) std::cout << "einv : " << i << std::endl;
	//}
	//for (size_t i = 0; i < mesh.getNumFaces(); i++) {
	//  if (i != mesh.faces[i]->id) std::cout << "f : " << i  << std::endl;
	//  if (mesh.faces[i]->isInvalid()) std::cout << "finv : " << i << std::endl;
	//}

public:

	template <typename SwapObj = std::vector<uint32_t>, typename SwapFunc = Swapper<typename SwapObj::value_type>>
	bool remove(ptv vert, SwapObj* swapOrder = nullptr) {
		// same as vert->isRemovable()
		auto&& ies = vert->incidentEdges<Ptr>();
		for (auto& e : ies) {
			if (!e->isBoundary() && e->pair->vert->isBoundary() && 
				!e->pair->next->isBoundary() && !e->prev()->isBoundary()) return false;
		}

		size_t i = 0;
		for (; i < (ies.size() - 2); i++) {
			remove<SwapObj, SwapFunc>(ies[i], swapOrder);
		}
		
		if (ies[i]->pair->face == nullptr) {
			remove<SwapObj, SwapFunc>(ies[i]->face, swapOrder);
		} else {
			if (remove<SwapObj, SwapFunc>(ies[i]->face, swapOrder)) {
				remove<SwapObj, SwapFunc>(ies[i]->pair->face, swapOrder);
			} else {
				remove<SwapObj, SwapFunc>(ies[i]->pair->face, swapOrder);
				remove<SwapObj, SwapFunc>(ies[i]->face, swapOrder);
			}
		}
		return true;
	}

	template <typename SwapObj = std::vector<uint32_t>, typename SwapFunc = Swapper<typename SwapObj::value_type>>
	bool remove(pte edge, SwapObj* swapOrder = nullptr) {
		if (edge->isBoundary()) { 
			return remove<SwapObj, SwapFunc>(edge->face, swapOrder);
		}
		
		if (!edge->isRemovable()) return false;

		pte pair = edge->pair;
		if (!edge->vert->isBoundary() || edge->vert->edge == edge) { edge->vert->edge = edge->next->pair; }
		if (!pair->vert->isBoundary() || pair->vert->edge == pair) { pair->vert->edge = pair->next->pair; }
		if (edge->face->edge == edge) { edge->face->edge = edge->next; }
		
		removeElement(pair->face);

		pte prev = pair;
		do { pair->face = edge->face; prev = pair; pair = pair->next; } while (pair != edge->pair);

		edge->prev()->next = pair->next;
		prev->next = edge->next;
				
		removeElement(edge);
		return true;
	}

	template <typename SwapObj = std::vector<uint32_t>, typename SwapFunc = Swapper<typename SwapObj::value_type>>
	bool remove(ptf face, SwapObj* swapOrder = nullptr) {
		if (!face->isRemovable()) return false;

		for (auto& e : face->circuitEdges<Ptr>()) {
			if (!e->vert->isBoundary() || e->vert->edge == e) e->vert->edge = e->next->pair;
			if (e->isBoundary()) {
				// remove unreferenced vertex
				if (e->vert->isUnrefer()) removeElement<ptv, SwapObj, SwapFunc>(e->vert, swapOrder);
				removeElement(e);
			} else {
				e->next = nullptr;
				e->face = nullptr;
				if (!e->isSecond()) {
					auto id = e->uid();
					edges[id] = edges[id]->pair;
					edges[id]->id = 2 * id;
					edges[id]->pair->id = 2 * id + 1;
				}
			}
		}

		removeElement(face);
		return true;
	}

	template <typename I = size_t>
	std::vector<I> removeUnrefVerts() {
		std::vector<I> swapOrder(getNumVerts());
		for (size_t i = 0; i < getNumVerts();) {
			if (verts[i]->isUnrefer()) { removeElement(verts[i], &swapOrder); } else { i++; }
		}
		return swapOrder;
	}

private:
	template <typename T>
	struct Inserter {
		void operator()(const uint32_t i, std::vector<T>& vec) {
			vec.emplace_back((T)i);
		}
	};

public:
	// edge split
	// @return inserted vertex index v2 | v1. (default std::numeric_limits<uint64_t>::max())
	template <typename InstObj = std::vector<uint32_t>, typename InstFunc = Inserter<typename InstObj::value_type>>
	uint64_t split(pte edge, InstObj* instOrder = nullptr) {
		if (edge->isBoundary()) return std::numeric_limits<uint64_t>::max();

		uint32_t s = (uint32_t)edges.size() * 2;

		auto pair = edge->pair;
		auto ei = edge->id;
		auto pi = pair->id;
		auto v = edge->vert;
		auto u = pair->vert;
		int insert = 0 + v->isBoundary() + 2 * u->isBoundary();
		if (edge->isSecond()) {
			edge->id = s;
			edges.emplace_back(edge);
		} else {
			pair->id = s;
			edges.emplace_back(pair);
		}

		edge->pair = make<Halfedge>(edge->id + 1);
		edge->pair->pair = edge;
		edge->pair->vert = u;

		pair->pair = make<Halfedge>(pair->id + 1);
		pair->pair->pair = pair;
		pair->pair->vert = v;

		uint64_t vi = std::numeric_limits<uint64_t>::max();

		auto insertAndRedirect = [this, &instOrder](auto& v) {
			auto vd = verts.emplace_back(make<Vertex>((uint32_t)verts.size()));
			if constexpr (!std::is_same_v<InstFunc, void>) InstFunc{}(v->id, *instOrder);
			auto ek = v->edge;
			do {
				ek->vert = vd;
				ek = ek->next->pair;
			} while (ek->face != nullptr);
			ek->vert = vd;
			vd->edge = v->edge;
		};

		if (insert == 1) {
			vi = v->id;
			insertAndRedirect(v);
		} else if (insert == 2) {
			vi = u->id;
			insertAndRedirect(u);
		} else if (insert == 3) {
			vi = ((uint64_t)u->id << 32) | v->id;
			insertAndRedirect(v);
			insertAndRedirect(u);
		}

		v->edge = edge;
		u->edge = pair;

		return vi;
	}

	// ----------------- mesh operation ---------------------

private:
	template <bool Triangulate = true>
	std::vector<std::vector<uint32_t>> closedFaces() const {
		auto bc = boundaryCycles<uint32_t>();
		auto flist = faceList<uint32_t>();

		for (const auto& cycle : bc) {
			if (cycle.size() < 3) continue;
			if constexpr (Triangulate) {
				uint32_t f0 = cycle[0], f1, f2;
				for (size_t i = 0; i < cycle.size() - 2; i++) {
					f1 = cycle[i+1];
					f2 = cycle[i+2];
					flist.emplace_back(std::vector<uint32_t>{ f0, f1, f2});
				}
			} else {
				flist.emplace_back(std::vector<uint32_t>(cycle.begin(), cycle.end()));
			}
		}
		return flist;
	}

public:

	// simple hole filling
	template <bool Triangulate = true>
	void fileHoles() {
		auto nv = getNumVerts();
		auto flist = closedFaces<Triangulate>();
		reset();
		build(nv, flist);
	}

	// return closed mesh (hole filling)
	template <bool Triangulate = true>
	HalfedgeMesh closedMesh() const {
		return HalfedgeMesh(getNumVerts(), closedFaces<Triangulate>());
	}

	// get dual faces list
	std::vector<std::vector<uint32_t>> dualFaceList() const {	
		uint32_t nf = (uint32_t)getNumFaces();
		std::vector<uint32_t> bface(edges.size(), 0);
		for (const auto& edge : edges) if (edge->isBoundary()) bface[edge->uid()] = nf++;

		std::vector<std::vector<uint32_t>> dfaces(getNumVerts());
		for (size_t i = 0; i < getNumVerts(); i++) {
			auto e = verts[i]->edge;

			if (e->isBoundary()) dfaces[i].emplace_back(bface[e->uid()]);
			do {
				dfaces[i].emplace_back(e->face->id);
				e = e->next->pair;
			} while (e->face != nullptr && e != verts[i]->edge);
			if (e->isBoundary()) dfaces[i].emplace_back(bface[e->uid()]);

			std::reverse(dfaces[i].begin(), dfaces[i].end());
		}
	
		return dfaces;
	}

	// get dual mesh
	// preserve prim->dual->prim(pdp) mesh boundaries : 
	// for (auto& face : pdp.faces) {
	// 	if (face->isBoundary()) prim.remove(face);
	// }
	HalfedgeMesh dualMesh() const {
		return HalfedgeMesh(getNumFaces() + getNumBoundaryEdges(), dualFaceList());
	}

	HalfedgeMesh triangleMesh() const {
		return HalfedgeMesh(getNumVerts(), util::faces2indices(faceList()));
	}

	template <typename T = uint32_t>
	std::pair<HalfedgeMesh, std::map<uint32_t, uint32_t>> mask(const std::vector<T>& maskFacesIndex) const {
		std::vector<uint32_t> maskVerts;
		std::map<uint32_t, uint32_t> vertexMap;

		std::vector<std::vector<uint32_t>> maskFaces(maskFacesIndex.size());
		for (size_t i = 0; i < maskFacesIndex.size(); i++) {
			maskFaces[i] = faces[maskFacesIndex[i]]->circuitVerts();
		}

		for (size_t i = 0, counter = 0; i < maskFaces.size(); i++) {
			for (auto& v : maskFaces[i]) {
				if (!vertexMap.contains(v)) {
					maskVerts.emplace_back(v);
					vertexMap[v] = counter++;
				}
				v = vertexMap[v];
			}
		}

		return { HalfedgeMesh(maskVerts.size(), maskFaces), vertexMap };
	}

};
}