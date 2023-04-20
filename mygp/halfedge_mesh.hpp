#pragma once
#include <bitset>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <queue>

#define _HEDS_RAW_PTR
// #define _HEDS_ABSEIL
// #define _HEDS_ROBIN
// #define _HEDS_RADIX

#ifdef _HEDS_ABSEIL
#include <absl/container/flat_hash_map.h>
#endif

#ifdef _HEDS_ROBIN
#include <robin_hood.h>
#endif

#ifdef _HEDS_RADIX
#include <parallel_radix_sort.h>
#endif

#include <mygp/geoutil.hpp>
#include <mygp/az_memory_pool.hpp>

/// 
/// type definition
/// 
namespace gp::heds {

class Vertex;
class Halfedge;
class Face;

#ifdef _HEDS_RAW_PTR
#define _hedsGetRawPtr(x) x
using ptv = Vertex*;
using pte = Halfedge*;
using ptf = Face*;
#else
#define _hedsGetRawPtr(x) x.get()
using ptv = std::shared_ptr<Vertex>;
using pte = std::shared_ptr<Halfedge>;
using ptf = std::shared_ptr<Face>;
#endif

using eKey = std::pair<uint32_t, uint32_t>;
inline eKey keyPair(const eKey& key) { return eKey(key.second, key.first); }
struct eKeyHash {
	uint64_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
		return static_cast<uint64_t>(p.first) << 32 | p.second;
	}
};

using ptVerts = std::vector<ptv>;
using ptFaces = std::vector<ptf>;
using ptEdges = std::vector<pte>;
using EKeyMap  = std::unordered_map<eKey, uint32_t, eKeyHash>;
//using EKeyMap = absl::flat_hash_map<eKey, uint32_t>;
//using EKeyMap = robin_hood::unordered_map<eKey, uint32_t, eKeyHash>;

/// for to search opposite edges
namespace impl {
template <typename T, typename I>
std::vector<std::pair<T, I>> zipVecIdx(const T* vec, const I* idx, const size_t size) {
	std::vector<std::pair<T, I>> vi(size);
	for (size_t i = 0; i < size; i++) {
		vi[i].first = vec[i];
		vi[i].second = idx[i];
	}
	return vi;
}

template <typename T, typename I>
std::pair<std::vector<T>, std::vector<I>> unzipVecIdx(const std::pair<T, I>* vi, const size_t size) {
	std::vector<T> vec(size);
	std::vector<I> idx(size);
	for (size_t i = 0; i < size; i++) {
		vec[i] = vi[i].first;
		idx[i] = vi[i].second;
	}
	return std::make_pair(vec, idx);
}

inline std::vector<uint32_t> sortEdgeKeys(std::vector<uint32_t>& left, std::vector<uint32_t>& right) {
	std::vector<uint32_t> idx(left.size());
	std::iota(idx.begin(), idx.end(), 0);

	#ifdef _HEDS_RADIX
	parallel_radix_sort::PairSort<uint32_t, std::pair<uint32_t, uint32_t>> pair_sort;
	pair_sort.Init(idx.size());

	auto vals = zipVecIdx(left.data(), idx.data(), idx.size());
	auto [sr, sli] = pair_sort.Sort(right.data(), vals.data(), idx.size());
	auto [sl, si] = unzipVecIdx(sli, left.size());
	vals = zipVecIdx(sr, si.data(), left.size());
	auto [ssl, ssri] = pair_sort.Sort(sl.data(), vals.data(), left.size());

	for (size_t i = 0; i < idx.size(); i++) {
		left[i] = ssl[i];
		right[i] = ssri[i].first;
		idx[i] = ssri[i].second;
	}

	#else
	std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> vals(left.size());
	for (uint32_t i = 0; i < vals.size(); i++) {
		std::get<0>(vals[i]) = left[i];
		std::get<1>(vals[i]) = right[i];
		std::get<2>(vals[i]) = i;
	}
	std::sort(vals.begin(), vals.end());
	for (size_t i = 0; i < vals.size(); i++) {
		left[i]  = std::get<0>(vals[i]);
		right[i] = std::get<1>(vals[i]);
		idx[i]	 = std::get<2>(vals[i]);
	}

	#endif
	return idx;
}

}
}

///
/// element data
/// 
namespace gp::heds {

class Vertex {
public:
	pte edge;

	uint32_t id;

	Vertex() : edge(nullptr), id(0) {}

	Vertex(const uint32_t id) : edge(nullptr), id(id) {}

	~Vertex() {}

	bool operator==(const Vertex& other) {
		return (id == other.id);
	}

	// get one-ring neighbors (ordered clockwise)
	ptVerts oneringNeighbs() const;

	// get one-ring neighbors (ordered clockwise)
	std::vector<uint32_t> oneringVindex() const;

	// get incident edges (ordered clockwise)
	ptEdges incidentEdges() const;

	// get incident edges (ordered clockwise)
	std::vector<uint32_t> incidentEindex() const;

	// get adjacent faces (ordered clockwise)
	ptFaces adjacentFaces() const;

	// get adjacent faces (ordered clockwise)
	std::vector<uint32_t> adjacentFindex() const;

	// get num of onerings
	size_t onerings() const;

	bool isBoundary() const;

};

class Halfedge {
public:
	pte next;
	pte pair;

	ptf face;
	ptv vert;

	uint32_t id;

	Halfedge() : next(nullptr), pair(nullptr), face(nullptr), vert(nullptr), id(0) {}
	Halfedge(const uint32_t id) : next(nullptr), pair(nullptr), face(nullptr), vert(nullptr), id(id) {}
	~Halfedge() {}

	// get adjacent faces
	std::pair<ptf, ptf> adjacentFaces() const;

	// get adjacent faces
	std::pair<uint32_t, uint32_t> adjacentFindex() const;

	// get endpoint vertex
	std::pair<ptv, ptv> endpoint() const;

	// get endpoint vertex
	std::pair<uint32_t, uint32_t> endpointIndex() const;

private:
	void contractSwitch();

public:

	// edge collapse (triangle only) 
	// @param vert : collapse to this vertex
	void contract(ptv& vert);

	// (triangle only)
	bool isContractible() const;

	// flip edge (triangle only)
	void flip();

	bool isBoundary() const;

};

class Face {
public:
	pte edge;
	uint32_t id;

	Face() : edge(nullptr), id(0) {}
	Face(const uint32_t id) : edge(nullptr), id(id) {}
	~Face() {}

	// get vertices (ordered counter-clockwise)
	ptVerts adjacentVerts() const;

	// get vertices (ordered counter-clockwise)
	std::vector<uint32_t> adjacentVindex() const;

	// get faces (ordered counter-clockwise)
	ptFaces adjacentFaces() const;

	// get faces (ordered counter-clockwise)
	std::vector<uint32_t> adjacentFindex() const;

	// (triangle only) 
	bool isInvalid() const;

	// (triangle only) 
	bool isBoundary() const;
};

}

/// 
/// element's functions
/// 
namespace gp::heds {
inline ptVerts Vertex::oneringNeighbs() const {
	ptVerts neighbs;
	neighbs.reserve(6);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		neighbs.emplace_back(e->pair->vert);
		e = _hedsGetRawPtr(e->next->pair);
	} while (e != e0 && e->face != nullptr);
	if (e->face == nullptr) neighbs.emplace_back(e->pair->vert);

	return neighbs;
}

inline ptEdges Vertex::incidentEdges() const {
	ptEdges edges;
	edges.reserve(6);
	pte e = this->edge;

	do {
		edges.emplace_back(e);
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) edges.emplace_back(e);

	return edges;
}

inline ptFaces Vertex::adjacentFaces() const {
	ptFaces faces;
	faces.reserve(6);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		faces.emplace_back(e->face);
		e = _hedsGetRawPtr(e->next->pair);
	} while (e != e0 && e->face != nullptr);

	return faces;
}

inline std::vector<uint32_t> Vertex::oneringVindex() const {
	std::vector<uint32_t> neighbs;
	neighbs.reserve(6);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		neighbs.emplace_back(e->pair->vert->id);
		e = _hedsGetRawPtr(e->next->pair);
	} while (e != e0 && e->face != nullptr);
	if (e->face == nullptr) neighbs.emplace_back(e->pair->vert->id);

	return neighbs;
}

inline std::vector<uint32_t> Vertex::incidentEindex() const {
	std::vector<uint32_t> edges;
	edges.reserve(6);
	pte e = this->edge;

	do {
		edges.emplace_back(e->id);
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) edges.emplace_back(e->id);

	return edges;
}

inline std::vector<uint32_t> Vertex::adjacentFindex() const {
	std::vector<uint32_t> faces;
	faces.reserve(6);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		faces.emplace_back(e->face->id);
		e = _hedsGetRawPtr(e->next->pair);
	} while (e != e0 && e->face != nullptr);

	return faces;
}

inline size_t Vertex::onerings() const {
	size_t c = 0;
	pte e = this->edge;

	do {
		c++;
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) c++;

	return c;
}

inline bool Vertex::isBoundary() const { return edge->pair->face == nullptr; }

inline std::pair<ptf, ptf> Halfedge::adjacentFaces() const {
	ptf i = face != nullptr ? face : nullptr;
	ptf j = pair->face != nullptr ? pair->face : nullptr;
	return std::make_pair(i, j);
}

inline std::pair<uint32_t, uint32_t> Halfedge::adjacentFindex() const {
	uint32_t i = face != nullptr ? face->id : std::numeric_limits<uint32_t>::max();
	uint32_t j = pair->face != nullptr ? pair->face->id : std::numeric_limits<uint32_t>::max();
	return std::make_pair(i, j);
}

inline void Halfedge::contractSwitch() {
	auto e1 = this->next;
	auto e2 = e1->next;
	if (e1->vert->edge == e1) e1->vert->edge = e2->pair;
	e1 = e1->pair;
	e2 = e2->pair;

	e1->pair = e2;
	e2->pair = e1;
}

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

inline void Halfedge::flip() {
	if (isBoundary()) return;
	pte a = next, b = a->next, c = pair->next, d = c->next;

	// trg vert's edge
	if (vert->edge == this) vert->edge = d;
	if (pair->vert->edge == pair) pair->vert->edge = b;

	// this -> trg vert
	vert = a->vert;
	pair->vert = c->vert;

	// face's edge
	face->edge = this;
	pair->face->edge = pair;

	// next->next's next
	b->next = c;
	d->next = a;

	// next's face
	a->face = pair->face;
	c->face = face;

	// next's next
	a->next = pair;
	c->next = this;

	// this -> next
	next = b;
	pair->next = d;
}

inline bool Halfedge::isBoundary() const {
	return face == nullptr || pair->face == nullptr;
}

inline std::pair<ptv, ptv> Halfedge::endpoint() const {
	return std::make_pair(pair->vert, vert);
}

inline std::pair<uint32_t, uint32_t> Halfedge::endpointIndex() const {
	return std::make_pair(pair->vert->id, vert->id);
}

inline ptVerts Face::adjacentVerts() const {
	ptVerts verts;
	verts.reserve(3);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		verts.emplace_back(e->vert);
		e = _hedsGetRawPtr(e->next);
	} while (e != e0);

	return verts;
}

inline ptFaces Face::adjacentFaces() const {
	ptFaces faces;
	faces.reserve(3);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		if (e->pair->face != nullptr) faces.emplace_back(e->pair->face);
		e = _hedsGetRawPtr(e->next);
	} while (e != e0);

	return faces;
}

inline std::vector<uint32_t> Face::adjacentVindex() const {
	std::vector<uint32_t> verts;
	verts.reserve(3);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		verts.emplace_back(e->vert->id);
		e = _hedsGetRawPtr(e->next);
	} while (e != e0);

	return verts;
}

inline std::vector<uint32_t> Face::adjacentFindex() const {
	std::vector<uint32_t> faces;
	faces.reserve(3);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		if (e->pair->face != nullptr) faces.emplace_back(e->pair->face->id);
		e = _hedsGetRawPtr(e->next);
	} while (e != e0);

	return faces;
}

inline bool Face::isInvalid() const {
	return this->edge != this->edge->pair->pair ||
		this->edge->next != this->edge->next->pair->pair ||
		this->edge->next->next != this->edge->next->next->pair->pair;
}

inline bool Face::isBoundary() const {
	return edge->pair->face == nullptr ||
		edge->next->pair->face == nullptr ||
		edge->next->next->pair->face == nullptr;
}

}

///  
/// mesh class
/// 
namespace gp::heds {
// halfedge mesh class
// requires gp::Is2DVec<Ptype> || gp::IsEigen<Ptype>
template <typename Ptype = std::vector<std::vector<double>>>
class Mesh {
	gp::util::AZMemoryPool<Vertex, Halfedge, Face> mpool;

	Ptype* refPos;

	template <typename T>
	auto make(const uint32_t i) {
		#ifdef _HEDS_RAW_PTR
		return new(mpool.bind<T>()) T(i);
		#else
		return std::shared_ptr<T>(new (mpool.bind<T>()) T(i), [](T* data) {});
		#endif
	}

	template <typename T>
	inline size_t vertexInputSize(const T& data) const {
		if constexpr (gp::util::IsEigen<T>) return static_cast<size_t>(data.rows());
		else if constexpr (gp::util::IsScalar<T>) return static_cast<size_t>(data);
		else return data.size();
	}

	template <typename T>
	inline size_t faceInputSize(const T& data) const {
		if constexpr (gp::util::IsEigen<T>) return static_cast<size_t>(data.rows());
		else if constexpr (gp::util::IsStdvec2d<T>) return static_cast<size_t>(data.size());
		else return static_cast<size_t>(data.size() / 3);
	}

	template <typename T>
	void initVertices(const T& vertexAttrib) {
		verts.resize(vertexInputSize(vertexAttrib));

		for (uint32_t i = 0; i < verts.size(); i++) {
			verts[i] = make<Vertex>(i);
		}
	}

	template <typename T>
	void initFaces(const T& faceAttrib) {
		uint32_t n = static_cast<uint32_t>(faceInputSize(faceAttrib)), edgeIndex = 0;

		faces.reserve(n);
		edges.reserve(3*n/2);

		#ifdef _HEDS_RADIX
		ptEdges halfedges;
		halfedges.reserve(3*n/2);
		std::vector<uint32_t> left, right;
		left.reserve(3*n/2); right.reserve(3*n/2);
		#else
		EKeyMap keymap;
		keymap.reserve(3 * n / 2);
		#endif

		std::vector<eKey> feKey; feKey.reserve(3);
		for (uint32_t i = 0; i < n; i++) {
			feKey.clear();
			if constexpr (gp::util::IsEigen<T>) {
				for (size_t j = 0, k = (size_t)faceAttrib.cols(); j < k; j++)
					feKey.emplace_back(faceAttrib(i, j), faceAttrib(i, (j+1)%k));
			} else if constexpr (gp::util::IsStdvec2d<T>) {
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

				#ifndef _HEDS_RADIX
				if (keymap.contains(keyPair(feKey[j]))) {
					edge = edges[keymap.at(keyPair(feKey[j]))]->pair;
				} else {
					keymap.emplace(feKey[j], edgeIndex/2);
					edge = edges.emplace_back(make<Halfedge>(edgeIndex++));
					edge->pair = make<Halfedge>(edgeIndex++);
					edge->pair->pair = edge;
					edge->pair->vert = first;
					edge->vert = second;
				}

				#else
				if (feKey[j].first < feKey[j].second) {
					left.emplace_back(feKey[j].first); right.emplace_back(feKey[j].second);
				} else {
					left.emplace_back(feKey[j].second); right.emplace_back(feKey[j].first);
				}
				edge = halfedges.emplace_back(make<Halfedge>(0));
				edge->vert = second;
				#endif

				if (j == 0) front = edge;
				else prev->next = edge;
				prev = edge;

				edge->face = faces[i];
				faces[i]->edge = edge;
				second->edge = edge;
			}

			if (front != nullptr) prev->next = front;
		}

		#ifdef _HEDS_RADIX
		edges.reserve(halfedges.size()/2);
		auto idx = impl::sortEdgeKeys(left, right);
		for (size_t i = 0; i < idx.size(); i++) {
			auto l1 = left[i+0], r1 = right[i+0],
				l2 = left[i+1], r2 = right[i+1];
			halfedges[idx[i+0]]->id = edgeIndex++;
			if (l1 == l2 && r1 == r2) {
				halfedges[idx[i+1]]->id = edgeIndex++;
				halfedges[idx[i+0]]->pair = halfedges[idx[i+1]];
				halfedges[idx[i+1]]->pair = halfedges[idx[i+0]];
				edges.emplace_back(halfedges[idx[i+0]]);
				i++;
			} else {
				auto edge = edges.emplace_back(halfedges[idx[i+0]]);
				edge->pair = make<Halfedge>(edgeIndex++);
				edge->pair->pair = edge;
				edge->pair->vert = verts[(edge->vert->id == l1) ? r1 : l1];
			}
		}
		#endif

	}

	void initBoundary() {
		for (const auto& e : edges) {
			if (e->pair->face == nullptr)
				verts[e->vert->id]->edge = e;
		}
	}

public:
	ptEdges edges;
	ptFaces faces;
	ptVerts verts;

	Mesh() : refPos(nullptr), edges({}), faces({}), verts({}) {}

	// halfedge data structure
	// @param vertexAttrib	: vertex matrix (2d)
	// @param faceAttrib	: face matrix (2d) or index list (1d)
	template<typename T1, class = std::void_t<decltype(std::declval<Ptype>()[0])>>
	Mesh(Ptype& vertexAttrib, const T1& faceAttrib) :
		refPos(nullptr), edges({}), faces({}), verts({}) {
		build(vertexAttrib, faceAttrib);
	}
	
	// halfedge data structure
	// @param n	: number of vertices (scalar)
	// @param faceAttrib	: face matrix (2d) or index list (1d)
	template<typename T1, typename T2>
	Mesh(const T1& n, const T2& faceAttrib) :
		refPos(nullptr), edges({}), faces({}), verts({}) {
		static_assert(std::is_scalar_v<T1>, 
			"HalfedgeMesh : vertexAttrib can only be lvalue pos or rvalue npos.");
		build(n, faceAttrib);
	}
	
	~Mesh() {
		reset(false);
	}

	// builds halfedge data structure
	// @param vertexAttrib	: vertex matrix (2d)
	// @param faceAttrib	: face matrix (2d) or index list (1d)
	template<typename T1, class = std::void_t<decltype(std::declval<Ptype>()[0])>>
	void build(Ptype& vertexAttrib, const T1& faceAttrib) {
		size_t nv = vertexInputSize(vertexAttrib), nf = faceInputSize(faceAttrib), ne = 3 * nf;
		mpool.ialloc(nv, ne, nf);
		
		bind(vertexAttrib);
		
		initVertices(vertexAttrib);
		initFaces(faceAttrib);
		initBoundary();
	}

	// builds halfedge data structure
	// @param n	: number of vertices (scalar)
	// @param faceAttrib	: face matrix (2d) or index list (1d)
	template<typename T1, typename T2>
	void build(const T1& n, const T2& faceAttrib) {
		size_t nv = static_cast<size_t>(n), nf = faceInputSize(faceAttrib), ne = 3 * nf;
		mpool.ialloc(nv, ne, nf);
		
		initVertices(nv);
		initFaces(faceAttrib);
		initBoundary();
	}

	void reset(bool resizeObject = true) {
		refPos = nullptr;

		for (auto& v : verts) v->~Vertex();
		for (auto& e : edges) {
			e->pair->~Halfedge();
			e->~Halfedge();
		}
		for (auto& f : faces) f->~Face();

		if (resizeObject) {
			verts = ptVerts{};
			faces = ptFaces{};
			edges = ptEdges{};
		}

		mpool.reset();
	}

	// UNDONE
	template <typename T>
	void changeConnectivity(const T& faceAttrib) {

	}

	// bind positions
	void bind(Ptype& bindPos) {
		refPos = &bindPos;
	}

	// bind positions
	void pbind(Ptype* bindPos) {
		refPos = bindPos;
	}

	Ptype* getPbind() {
		return refPos;
	}

	// ----------------- mesh information ---------------------

	size_t getNumVerts() const {
		return verts.size();
	}

	size_t getNumFaces() const {
		return faces.size();
	}

	size_t getNumEdges() const {
		return edges.size();
	}

	int eulerPoincare() const {
		return (int)getNumVerts() - (int)getNumEdges() + (int)getNumFaces();
	}

	pte halfedge(const uint32_t s, const uint32_t t) const {
		pte edge = nullptr;
		for (const auto& e : verts[s]->incidentEdges()) {
			if (e->pair->vert->id == t) {
				edge = e;
				break;
			}
		}

		return edge;
	}

	template <typename T1, typename T2>
	pte halfedge(const std::pair<T1, T2>& e) const {
		return halfedge((uint32_t)e.first, (uint32_t)e.second);
	}

	Ptype copyPosition() const {
		return refPos[0];
	}

	void swapPosition(const size_t i, const size_t j) {
		if constexpr (gp::util::IsEigen<Ptype>)
			refPos->row(i).swap(refPos->row(j));
		else
			std::swap(refPos[0][i], refPos[0][j]);
	}

	void resizePosition(const size_t size) {
		if constexpr (gp::util::IsEigen<Ptype>)
			refPos[0].conservativeResize(size, refPos[0].cols());
		else
			refPos[0].resize(size);
	}

	template<typename T>
	void setPosition(const size_t i, const T& pos) {
		if constexpr (gp::util::IsEigen<Ptype>) {
			refPos[0](i, 0) = pos[0];
			refPos[0](i, 1) = pos[1];
			refPos[0](i, 2) = pos[2];
		} else {
			refPos[0][i][0] = pos[0];
			refPos[0][i][1] = pos[1];
			refPos[0][i][2] = pos[2];
		}
	}

	template<typename T>
	ptv appendVertex(const T& pos) {
		ptv p = make<Vertex>();

		if constexpr (gp::util::IsEigen<Ptype>) {
			auto i = refPos[0].rows();
			p->id = (uint32_t)i;
			refPos[0].conservativeResize(i+1, refPos[0].cols());
			refPos[0](i, 0) = pos[0];
			refPos[0](i, 1) = pos[1];
			refPos[0](i, 2) = pos[2];
		} else {
			p->id = (uint32_t)refPos[0].size();
			refPos[0].emplace_back(T(pos[0], pos[1], pos[2]));
		}

		verts.emplace_back(p);
		return p;
	}

	// vertex adjacent list
	template <typename T = uint32_t>
	std::vector<std::vector<T>> adjacencyList(const bool self = false) const {
		std::vector<std::vector<T>> adlist(this->getNumVerts());

		for (size_t i = 0; i < this->getNumVerts(); i++) {
			for (const auto& v: this->verts[i]->oneringVindex()) adlist[i].emplace_back(static_cast<T>(v));
			if (self) adlist[i].emplace_back(static_cast<T>(i));
		}
		return adlist;
	}

	// face index list
	template <typename T=uint32_t>
	std::vector<std::vector<T>> faceIndexList() const {
		std::vector<std::vector<T>>	F(getNumFaces());

		for (size_t i = 0; i < getNumFaces(); i++) {
			for (const auto& v : faces[i]->adjacentVindex())
				F[i].emplace_back(static_cast<T>(v));
		}
		return F;
	}

	template <typename T = Ptype::value_type, typename F = T::value_type>
	T position(const size_t i) const {
		return refPos[0][i];
	}

	template <typename T = Ptype::value_type, typename F = T::value_type>
	T edge2vec(const pte& edge) const {
		T e0 = position(edge->pair->vert->id);
		T e1 = position(edge->vert->id);
		return { e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2] };
	}
		
	template <typename T = size_t>
	std::vector<T> countVertexIncidents() const {
		std::vector<T> vi(getNumVerts(), 0);
		for (const auto& edge : edges) {
			vi[edge->vert->id]++;
			vi[edge->pair->vert->id]++;
		}
		return vi;
	}

	std::vector<size_t> rRingNeighbs(const size_t src, const double radius) const {
		std::vector<size_t> neighbs;

		using node = std::pair<double, size_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		search.emplace(0.0, src);
		std::vector<double> minDist(getNumVerts(), std::numeric_limits<double>::max());
		minDist[src] = 0.0;
		std::vector<bool> used(getNumVerts(), false);
		used[src] = true;

		while (!search.empty()) {
			auto [d, v] = search.top();
			search.pop();

			for (const auto& u : verts[v]->oneringVindex()) {
				double dist = d + (position(u) - position(v)).norm();
				if (dist < minDist[u] && dist <= radius) {
					search.emplace(dist, u);
					minDist[u] = dist;
					if (used[u]) continue;
					neighbs.emplace_back(u);
					used[u] = true;
				}
			}
		}

		return neighbs;
	}

	std::vector<size_t> kRingNeighbs(const size_t src, const size_t k) const {
		std::vector<size_t> neighbs;

		std::deque<std::pair<size_t, size_t>> search;
		search.emplace_back(0, src);
		std::vector<bool> used(getNumVerts(), false);
		used[src] = true;

		while (!search.empty()) {
			auto [d, v] = search.front();
			search.pop_front();

			if (d == k) continue;
			for (const auto& u : verts[v]->oneringVindex()) {
				if (!used[u]) {
					search.emplace_back(d + 1, u);
					neighbs.emplace_back(u);
					used[u] = true;
				}
			}
		}

		return neighbs;
	}

	// get boundary cycles (counter-clockwise)
	template <typename T=size_t>
	std::vector<std::vector<T>> boundaryCycles() const {
		std::vector<std::vector<T>> boundaries;
		std::vector<bool> inCycles(getNumEdges(), false);

		for (const auto& edge : edges) {
			if (!edge->isBoundary() || inCycles[edge->id / 2]) continue;

			boundaries.push_back({});

			auto e0 = edge, e = e0;
			do {
				boundaries.back().emplace_back((T)e->vert->id);
				inCycles[e->id / 2] = true;

				e = e->next;
				while (!e->isBoundary()) {
					e = e->pair->next;
				};
			} while (e != e0);

			std::reverse(boundaries.back().begin(), boundaries.back().end());
		}

		return boundaries;
	}

	// --------------- geometry ----------------------
	
	// face center
	template <typename T = Ptype::value_type, typename F = T::value_type>
	T center(const ptf& face) const {
		T c;

		if constexpr (gp::util::IsStdvec<T>) c.resize(3);
		for (size_t i = 0; i < 3; i++) c[i] = 0;

		auto&& verts = face->adjacentVindex();
		for (const auto& v : verts) {
			c[0] += refPos[0][v][0];
			c[1] += refPos[0][v][1];
			c[2] += refPos[0][v][2];
		}

		for (size_t i = 0; i < 3; i++) c[i] /= verts.size();

		return c;
	}

	// face normal (unnormalized)
	template <typename T = Ptype::value_type, typename F = T::value_type>
	T normal(const ptf& face) const {
		auto i = face->edge->vert->id;
		auto j = face->edge->next->vert->id;
		auto k = face->edge->next->next->vert->id;
		
		T a = refPos[0][i], b = refPos[0][j], c = refPos[0][k];
		return { (b[1] - a[1])* (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
						(b[2] - a[2])* (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
						(b[0] - a[0])* (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) };
	}

	// vertex normal (normalized)
	template <typename T = Ptype::value_type, typename F = T::value_type>
	T normal(const ptv& vert) const {
		T n{ 0.0, 0.0, 0.0 };

		for (const auto& face : vert->adjacentFaces()) {
			auto norm = normal(face);
			n[0] += norm[0];
			n[1] += norm[1];
			n[2] += norm[2];
		}

		double norm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
		for (int i = 0; i < 3; i++) n[i] /= norm;
		return n;
	}

	// edge length
	double length(const pte& edge) const {
		if (refPos == nullptr) { throw std::runtime_error("positions are not valid"); }

		auto i = edge->pair->vert->id, j = edge->vert->id;
		if constexpr (gp::util::IsEigen<Ptype>) {
			return (refPos->row(i) - refPos->row(j)).norm();
		} else {
			double s0 = refPos[0][i][0], s1 = refPos[0][i][1], s2 = refPos[0][i][2],
				t0 = refPos[0][j][0], t1 = refPos[0][j][1], t2 = refPos[0][j][2];
			return sqrt(util::pow2(t0 - s0) + util::pow2(t1 - s1) + util::pow2(t2 - s2));
		}
	}

	// face area
	double area(const ptf& face) const {
		if constexpr (gp::util::IsEigen<Ptype>) {
			return 0.5 * normal(face).norm();
		} else {
			auto n = normal(face);
			return 0.5 * sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
		}
	}

	// average edge length
	double avgEdgeLength() const {
		double s = 0.0;
		for (const auto& edge : edges) {
			s += length(edge);
		}
		s /= (double)edges.size();
		return s;
	}

	// surface area
	double surfaceArea() const {
		double s = 0.0;
		for (const auto& face : faces) {
			s += area(face);
		}
		return s;
	}

	// dot product between e0 and e1
	double dot(const pte& e0, const pte& e1) const {
		auto i = e0->vert->id, j = e0->pair->vert->id;
		auto k = e1->vert->id, r = e1->pair->vert->id;
		if constexpr (gp::util::IsEigen<Ptype>) {
			return (refPos->row(i) - refPos->row(j)).dot(refPos->row(k) - refPos->row(r));
		} else {
			return (refPos[0][i][0] - refPos[0][j][0]) * (refPos[0][k][0] - refPos[0][r][0]) +
				(refPos[0][i][1] - refPos[0][j][1]) * (refPos[0][k][1] - refPos[0][r][1]) +
				(refPos[0][i][2] - refPos[0][j][2]) * (refPos[0][k][2] - refPos[0][r][2]);
		}
	}

	// cotanget correspondence to edge
	double cot(const pte& edge) const {
		auto i = edge->pair->vert->id;
		auto j = edge->vert->id;
		auto k = edge->next->vert->id;

		if constexpr (gp::util::IsEigen<Ptype>) {
			return gp::util::cot(refPos->row(k), refPos->row(i), refPos->row(j));
		} else {
			return gp::util::cot(refPos[0][k], refPos[0][i], refPos[0][j]);
		}
	}

	// angle between e0 and e1
	double angle(const pte& e0, const pte& e1) const {
		auto i = e0->vert->id, j = e0->pair->vert->id;
		auto k = e1->vert->id, r = e1->pair->vert->id;
		if constexpr (gp::util::IsEigen<Ptype>) {
			#ifdef EIGEN_MAJOR_VERSION
			return gp::util::angle(refPos->row(i) - refPos->row(j), refPos->row(k) - refPos->row(r));
			#endif
		} else {
			double ijx = refPos[0][i][0] - refPos[0][j][0];
			double ijy = refPos[0][i][1] - refPos[0][j][1];
			double ijz = refPos[0][i][2] - refPos[0][j][2];
			double krx = refPos[0][k][0] - refPos[0][r][0];
			double kry = refPos[0][k][1] - refPos[0][r][1];
			double krz = refPos[0][k][2] - refPos[0][r][2];
			double d = ijx * krx + ijy * kry + ijz * krz;
			double ijn = sqrt(ijx * ijx + ijy * ijy + ijz * ijz);
			double krn = sqrt(krx * krx + kry * kry + krz * krz);
			return acos(std::min(std::max(d / (ijn * krn), -1.0), 1.0));
		}
	}

	bool isObtuseTriangle(const ptf& tri) const {
		auto& e0 = tri->edge;
		auto& e1 = e0->next;
		auto& e2 = e1->next;

		double a = length(e0), b = length(e1), c = length(e2);
		double a2 = a * a, b2 = b * b, c2 = c * c;
		return gp::util::isObtuseTriangle(a, b, c);
	}

	// get dual mesh (TODO : boundary preserve)
	template <typename I = uint32_t, typename T = Ptype::value_type, typename F = T::value_type>
	std::pair<Ptype, std::vector<std::vector<I>>> dual() const {
		Ptype dverts(getNumFaces());
		for (size_t i = 0; i < getNumFaces(); i++) {
			dverts[i] = center(faces[i]);
		}

		std::vector<std::vector<I>> dfaces;
		for (size_t i = 0; i < getNumVerts(); i++) {
			const auto& fs = verts[i]->adjacentFindex();
			if (fs.size() <= 2) continue;
			dfaces.emplace_back(std::vector<I>());
			for (auto itr = fs.rbegin(); itr != fs.rend(); ++itr) {
				dfaces.back().emplace_back(static_cast<I>(*itr));
			}
		}

		return std::make_pair(dverts, dfaces);
	}

	void holeFilling() {
		auto bc = boundaryCycles<uint32_t>();
		auto faceList = faceIndexList<uint32_t>();

		for (const auto& cycle : bc) {
			if (cycle.size() < 3) continue;
			uint32_t f0 = cycle[0], f1, f2;
			for (size_t i = 0; i < cycle.size() - 2; i++) {
				f1 = cycle[i+1];
				f2 = cycle[i+2];
				faceList.emplace_back(std::vector<uint32_t>{ f0, f1, f2});
			}
		}

		auto ptr = getPbind();
		reset();
		build(*ptr, faceList);
	}

	bool isDelaunay() {
		for (const auto& edge : edges) {
			if (edge->pair->face == nullptr) continue;

			double alpha = angle(edge->next->pair, edge->next->next);
			double beta = angle(edge->pair->next->pair, edge->pair->next->next);

			if ((alpha + beta) > std::numbers::pi) return false;
		}
		return true;
	}

	// do edge flip while mesh isn't delaunay triangles
	void delaunayEdgeFlip() {
		bool flip = true;
		while (flip) {
			flip = false;
			for (auto& edge : edges) {
				if (edge->pair->face == nullptr) continue;

				double alpha = angle(edge->next->pair, edge->next->next);
				double beta = angle(edge->pair->next->pair, edge->pair->next->next);

				if ((alpha + beta) > std::numbers::pi) {
					edge->flip();
					flip = true;
				}
			}
		}
	}

	// get delaunay mesh
	Mesh<Ptype> delaunayMesh() {
		Mesh<Ptype> delaunay;
		delaunay.build(*refPos, faceIndexList());
		delaunay.delaunayEdgeFlip();
		return delaunay;
	}

	// ---------- return eigen vector version -----------
	#ifdef EIGEN_MAJOR_VERSION
	template <typename T = Ptype::value_type, typename = std::enable_if_t<std::is_scalar_v<T>>>
	Eigen::Vector3<T> position(const size_t i) const {
		return refPos->row(i);
	}

	template <typename T = Ptype::value_type, typename = std::enable_if_t<std::is_scalar_v<T>>>
	Eigen::Vector3<T> edge2vec(const pte& edge) const {
		return position(edge->vert->id) - position(edge->pair->vert->id);
	}

	template <typename T = Ptype::value_type, typename = std::enable_if_t<std::is_scalar_v<T>>>
	Eigen::Vector3<T> center(const ptf& face) const {
		return refPos[0](face->adjacentVindex(), Eigen::all).colwise().mean();
	}

	template <typename T = Ptype::value_type, typename = std::enable_if_t<std::is_scalar_v<T>>>
	Eigen::Vector3<T> normal(const ptf& face) const {
		//auto i = face->edge->vert->id;
		//auto j = face->edge->next->vert->id;
		//auto k = face->edge->next->next->vert->id;
		//
		//return (refPos->row(j) - refPos->row(i)).cross(refPos->row(k) - refPos->row(i));

		Eigen::Vector3<T> n = Eigen::Vector3<T>::Zero();
		auto&& vidx = face->adjacentVindex();
		for (size_t i = 0, k = vidx.size(); i < k; i++) {
			n += refPos->row(vidx[i]).cross(refPos->row(vidx[(i+1)%k]));
		}
		//n *= 0.5; // area function do 0.5 times
		return n.array();
	}

	template <typename T = Ptype::value_type, typename = std::enable_if_t<std::is_scalar_v<T>>>
	Eigen::Vector3<T> normal(const ptv& vert) const {
		Eigen::Vector3<T> n = Eigen::Vector3<T>::Zero();

		for (const auto& face : vert->adjacentFaces()) {
			n += normal(face);
		}

		return n.normalized();
	}

	template <typename I = int>
	Eigen::MatrixX<I> getFaceIndexMat() const {
		Eigen::MatrixX<I>	F(getNumFaces(), 3);
		for (size_t i = 0; i < getNumFaces(); i++) {
			auto&& v = faces[i]->adjacentVindex();
			F(i, 0) = static_cast<I>(v[0]);
			F(i, 1) = static_cast<I>(v[1]);
			F(i, 2) = static_cast<I>(v[2]);
		}
		return F;
	}

	template <typename T = Ptype::value_type, typename = std::enable_if_t<std::is_scalar_v<T>>>
	Eigen::MatrixX3<T> vertexNormals() const {
		Eigen::MatrixX3<T> normals = Eigen::MatrixX3<T>::Zero(getNumVerts(), 3);
		for (size_t i = 0; i < getNumFaces(); i++) {
			auto n = normal(faces[i]);
			for (const auto& v : faces[i]->adjacentVindex()) {
				normals.row(v) += n.transpose();
			}
		}
		normals.rowwise().normalize();
		return normals;
	}

	// get dual mesh (TODO : boundary preserve)
	template <typename I = uint32_t, typename F = Ptype::value_type, 
		typename = std::enable_if_t<std::is_scalar_v<F>>>
	std::pair<Eigen::MatrixX3<F>, std::vector<std::vector<I>>> dual() const {
		Eigen::MatrixX3<F> dverts(getNumFaces(), 3);
		for (size_t i = 0; i < getNumFaces(); i++) {
			dverts.row(i) = center(faces[i]);
		}

		std::vector<std::vector<I>> dfaces;
		for (size_t i = 0; i < getNumVerts(); i++) {
			const auto& fs = verts[i]->adjacentFindex();
			dfaces.emplace_back(std::vector<I>());
			if (fs.size() <= 2) continue;
			for (auto itr = fs.rbegin(); itr != fs.rend(); ++itr) {
				dfaces.back().emplace_back(static_cast<I>(*itr));
			}
		}

		return std::make_pair(dverts, dfaces);
	}

	// normalize binded positions
	void normalize(const bool normWithSurfaceArea = false) {
		refPos->array().rowwise() -= refPos->colwise().mean().array();
		if (normWithSurfaceArea)
			refPos->array() /= surfaceArea();
		else
			refPos->array() /= refPos->maxCoeff();
	}

	// check point p is inside mesh 
	// using winding number (check all solid angle)
	bool isInside(const Eigen::Vector3d& p, const double eps = 1e-8) {
		double angle = 0.0;
		for (size_t i = 0; i < getNumFaces(); i++) {
			Eigen::Vector3d a = position(faces[i]->edge->vert->id);
			Eigen::Vector3d b = position(faces[i]->edge->next->vert->id);
			Eigen::Vector3d c = position(faces[i]->edge->pair->vert->id);
			angle += gp::util::solidAngle(a - p, b - p, c - p);
		}

		return !(angle < eps);
	}
	#endif

};

}