#pragma once
#include <bitset>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __has_include("glm/glm.hpp")
#include <glm/glm.hpp>
#endif

#if __has_include("Eigen/Eigen")
#include <Eigen/Core>
#include <Eigen/Dense>
#endif

#ifdef GLM_ENABLE
#define USE_GLM
#endif

#ifdef EIGEN_CORE_H
#define USE_EIGEN
#endif

#ifdef _HEDS_ABSEIL
#include <absl/container/flat_hash_map.h>
#endif

#include <mygp/geoutil.hpp>
#include <mygp/az_memory_pool.hpp>

#define _HEDS_RAW_PTR
//#define _HEDS_ABSEIL
//#define _HEDS_VECTOR

namespace gp::heds {
/// type definition

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
#if defined(_HEDS_ABSEIL)
using ptEdges = absl::flat_hash_map<eKey, pte>;
#define _heds_acc .second
#elif defined(_HEDS_VECTOR)
using ptEdges = std::vector<pte>;
#define _heds_acc 
#else
using ptEdges = std::unordered_map<eKey, pte, eKeyHash>;
#define _heds_acc .second
#endif

}

namespace gp::heds {
/// element data

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

	// get incident edges (ordered clockwise)
	std::vector<pte> incidentEdges() const;

	// get adjacent faces (ordered clockwise)
	ptFaces adjacentFaces() const;

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
	std::pair<uint32_t, uint32_t> adjacentFaces() const;

private:
	void contractSwitch();

public:

	// contract this edge (triangle only)
	void contract(ptv& vert);

	bool isContractible() const;

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

	bool isInvalid() const;
};

}

namespace gp::heds {
ptVerts Vertex::oneringNeighbs() const {
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

std::vector<pte> Vertex::incidentEdges() const {
	std::vector<pte> edges;
	edges.reserve(6);
	pte e = this->edge;

	do {
		edges.emplace_back(e);
		e = e->next->pair;
	} while (e != this->edge && e->face != nullptr);
	if (e->face == nullptr) edges.emplace_back(e);

	return edges;
}

ptFaces Vertex::adjacentFaces() const {
	ptFaces faces;
	faces.reserve(6);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		faces.emplace_back(e->face);
		e = _hedsGetRawPtr(e->next->pair);
	} while (e != e0 && e->face != nullptr);

	return faces;
}

bool Vertex::isBoundary() const { return edge->pair->face == nullptr; }

std::pair<uint32_t, uint32_t> Halfedge::adjacentFaces() const {
	uint32_t i = face != nullptr ? face->id : std::numeric_limits<uint32_t>::max();
	uint32_t j = pair->face != nullptr ? pair->face->id : std::numeric_limits<uint32_t>::max();
	return std::pair<uint32_t, uint32_t>(i, j);
}

void Halfedge::contractSwitch() {
	auto e1 = this->next;
	auto e2 = e1->next;
	if (e1->vert->edge == e1) e1->vert->edge = e2->pair;
	e1 = e1->pair;
	e2 = e2->pair;

	e1->pair = e2;
	e2->pair = e1;
}

void Halfedge::contract(ptv& vert) {
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

bool Halfedge::isContractible() const {
	if (!isBoundary() && vert->isBoundary() && pair->vert->isBoundary()) return false;

	int count = isBoundary();
	auto&& oring = vert->oneringNeighbs();
	std::unordered_set<ptv> set;
	set.reserve(oring.size());
	for (const auto& v : oring) set.emplace(v);
	for (const auto& v : pair->vert->oneringNeighbs()) count += set.contains(v);
	return count < 3;
}

bool Halfedge::isBoundary() const {
	return face == nullptr || pair->face == nullptr;
}

ptVerts Face::adjacentVerts() const {
	ptVerts verts;
	verts.reserve(3);
	Halfedge* e0 = _hedsGetRawPtr(this->edge), * e = e0;

	do {
		verts.emplace_back(e->vert);
		e = _hedsGetRawPtr(e->next);
	} while (e != e0);

	return verts;
}

bool Face::isInvalid() const {
	return this->edge != this->edge->pair->pair ||
		this->edge->next != this->edge->next->pair->pair ||
		this->edge->next->next != this->edge->next->next->pair->pair;
}

}



namespace gp::heds {
// halfedge mesh class
// requires gp::Is2DVec<Ptype> || gp::IsEigen<Ptype>
template <typename Ptype = std::vector<std::vector<double>>>
class Mesh {

	gp::util::AZMemoryPool<Vertex, Halfedge, Face> mpool;

	Ptype* refPos;

	template <typename T>
	inline size_t vertexInputSize(T& data) {
		if constexpr (gp::IsEigen<T>) return static_cast<size_t>(data.rows());
		else if constexpr (gp::IsScalar<T>) return static_cast<size_t>(data);
		else return data.size();
	}

	template <typename T>
	inline size_t faceInputSize(T& data) {
		if constexpr (gp::IsEigen<T>) return static_cast<size_t>(data.rows());
		else if constexpr (gp::Is2DVec<T>) return static_cast<size_t>(data.size());
		else return static_cast<size_t>(data.size() / 3);
	}

	template <typename T>
	inline auto make(uint32_t i) {
		#ifdef _HEDS_RAW_PTR
		return new(mpool.bind<T>()) T(i);
		#else
		return std::shared_ptr<T>(new (mpool.bind<T>()) T(i), [](T* data) {});
		#endif
	}

	template <typename T>
	void initVertices(T& positions) {
		verts.resize(vertexInputSize(positions));

		for (uint32_t i = 0; i < verts.size(); i++) {
			verts[i] = make<Vertex>(i);
		}
	}

	template <typename T>
	void initFaces(const T& faceIndices) {
		uint32_t n = static_cast<uint32_t>(faceInputSize(faceIndices)), edgeIndex = 0;

		faces.reserve(n);
		edges.reserve(3*n/2);
		#if defined(_HEDS_VECTOR)
		std::unordered_map<uint64_t, uint32_t> keys;
		keys.reserve(3*n/2);
		#endif
		std::vector<eKey> feKey; feKey.reserve(3);
		for (uint32_t i = 0; i < n; i++) {
			feKey.clear();
			if constexpr (gp::IsEigen<T>) {
				for (size_t j = 0, k = (size_t)faceIndices.cols(); j < k; j++)
					feKey.emplace_back(faceIndices(i, j), faceIndices(i, (j+1)%k));
			} else if constexpr (gp::Is2DVec<T>) {
				for (size_t j = 0, k = faceIndices[i].size(); j < k; j++)
					feKey.emplace_back(faceIndices[i][j], faceIndices[i][(j+1)%k]);
			} else {
				feKey.emplace_back(faceIndices[i * 3 + 0], faceIndices[i * 3 + 1]);
				feKey.emplace_back(faceIndices[i * 3 + 1], faceIndices[i * 3 + 2]);
				feKey.emplace_back(faceIndices[i * 3 + 2], faceIndices[i * 3 + 0]);
			}

			assert(feKey.size() != 0);

			faces.emplace_back(make<Face>(i));

			pte front = nullptr, prev = nullptr;
			for (size_t j = 0; j < feKey.size(); j++) {
				auto& first = verts[feKey[j].first], &second = verts[feKey[j].second];
				
				pte edge;
				if (edges.contains(keyPair(feKey[j]))) {
					edge = edges.at(keyPair(feKey[j]))->pair;
				} else {
					auto&& ret = edges.emplace(feKey[j], make<Halfedge>(edgeIndex++));
					edge = (*ret.first).second;
					edge->pair = make<Halfedge>(edgeIndex++);
					edge->pair->pair = edge;
					edge->vert = second;
					edge->pair->vert = first;
				}

				if (j == 0) front = edge;
				else prev->next = edge;
				prev = edge;
				edge->face = faces[i];
				faces[i]->edge = edge;
				second->edge = edge;
			}
			prev->next = front;
		}
	}

	void initBoundary() {
		for (const auto& e : edges) {
			if (e _heds_acc->pair->face == nullptr)
				verts[e _heds_acc->vert->id]->edge = e _heds_acc;
		}
	}

public:
	ptEdges edges;
	ptFaces faces;
	ptVerts verts;

	Mesh() : refPos(nullptr), edges({}), faces({}), verts({}) {}

	// halfedge data structure
	// @param positions	: vertex matrix (2d)
	// @param faceIndices	: face matrix (2d) or index list (1d)
	template<typename T1>
	Mesh(Ptype& positions, const T1& faceIndices) :
		refPos(nullptr), edges({}), faces({}), verts({}) {
		build(positions, faceIndices);
	}

	// halfedge data structure
	// @param n		: number of vertices (scalar)
	// @param faceIndices	: face matrix (2d) or index list (1d)
	template<gp::IsScalar T1, typename T2>
	Mesh(const T1& n, const T2& faceIndices) :
		refPos(nullptr), edges({}), faces({}), verts({}) {
		build(n, faceIndices);
	}

	~Mesh() {
		reset(false);
	}

	size_t getNumVerts() const {
		return verts.size();
	}

	size_t getNumFaces() const {
		return faces.size();
	}

	size_t getNumEdges() const {
		return edges.size();
	}

	pte halfedge(const uint32_t s, const uint32_t t) const {
		if (edges.contains(eKey(s, t))) return edges.at(eKey(s, t));
		else if (edges.contains(eKey(t, s))) return edges.at(eKey(t, s))->pair;
		return nullptr;
	}

	// builds halfedge data structure
	// @param positions	: vertex matrix (2d)
	// @param faceIndices	: face matrix (2d) or index list (1d)
	template<typename T1>
	void build(Ptype& positions, const T1& faceIndices) {
		size_t nv = vertexInputSize(positions), nf = faceInputSize(faceIndices), ne = 3 * nf;
		mpool.ialloc(nv, ne, nf);

		bind(positions);

		initVertices(positions); 
		initFaces(faceIndices);
		initBoundary();
	}

	// builds halfedge data structure
	// @param n		: number of vertices (scalar)
	// @param faceIndices	: face matrix (2d) or index list (1d)
	template<gp::IsScalar T1=size_t, typename T2=std::vector<std::vector<int>>>
	void build(const T1& n, const T2& faceIndices) {
		size_t nv = static_cast<size_t>(n), nf = faceInputSize(faceIndices), ne = 3 * nf;
		mpool.ialloc(nv, ne, nf);

		initVertices(nv);
		initFaces(faceIndices);
		initBoundary();
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

	void reset(bool resizeObject = true) {
		refPos = nullptr;

		for (auto& v : verts) v->~Vertex();
		for (auto& e : edges) {
			e _heds_acc->pair->~Halfedge();
			e _heds_acc->~Halfedge();
		}
		for (auto& f : faces) f->~Face();

		if (resizeObject) {
			verts = ptVerts{};
			faces = ptFaces{};
			edges = ptEdges{};
		}

		mpool.reset();
	}

	//TODO
	void rebuildEdges() {
		//edges.clear();
		//
		//for (const auto& face : this->faces) {
		//	auto e0 = face->edge;
		//	auto e = face->edge;
		//	do {
		//		eKey key(e->vert->id, e->pair->vert->id);
		//		if (!edges.contains(keyPair(key))) edges[key] = e;
		//		e = e->next;
		//	} while (e0 != e);
		//}
		//
		//for (uint32_t i = 0; auto& emap : edges) {
		//	emap.second->id = i++;
		//	emap.second->pair->id = i++;
		//}
	}

	// --------------------------------------------------------------------

	Ptype copyPosition() const {
		return refPos[0];
	}

	template<typename Ret>
	Ret getPosition(const size_t i) const {
		if constexpr (gp::IsEigen<Ptype> && gp::IsEigen<Ret>)
			return refPos->row(i);
		else {
			Ret p;
			if constexpr (gp::IsVec<Ret>) p.resize(3);

			p[0] = refPos[0][i][0];
			p[1] = refPos[0][i][1];
			p[2] = refPos[0][i][2];
			return p;
		}
	}

	void swapPosition(const size_t i, const size_t j) {
		if constexpr (gp::IsEigen<Ptype>)
			refPos->row(i).swap(refPos->row(j));
		else
			std::swap(refPos[0][i], refPos[0][j]);
	}

	void resizePosition(const size_t size) {
		if constexpr (gp::IsEigen<Ptype>)
			refPos[0].conservativeResize(size, refPos[0].cols());
		else
			refPos[0].resize(size);
	}

	template<typename T>
	void setPosition(const size_t i, const T& pos) {
		if constexpr (gp::IsEigen<Ptype>) {
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
	ptv appendPosition(const T& pos) {
		ptv p = make<Vertex>();

		if constexpr (gp::IsEigen<Ptype>) {
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

	std::vector<std::vector<uint32_t>> adjacencyList() const {
		std::vector<std::vector<uint32_t>> adlist(this->getNumVerts());

		for (size_t i = 0; i < this->getNumVerts(); i++) {
			for (const auto& v: this->verts[i]->oneringNeighbs()) adlist[i].emplace_back(v->id);
		}
		return adlist;
	}

	template <typename T=uint32_t>
	std::vector<std::vector<T>> faceIndexList() {
		std::vector<std::vector<T>>	F(getNumFaces());

		for (size_t i = 0; i < getNumFaces(); i++) {
			for (const auto& v : faces[i]->adjacentVerts())
				F[i].emplace_back(static_cast<T>(v->id));
		}
		return F;
	}

	#ifdef USE_EIGEN
	Eigen::MatrixXi getFaceIndexMat() {
		Eigen::MatrixXi	F(getNumFaces(), 3);
		for (size_t i = 0; i < getNumFaces(); i++) {
			auto v = faces[i]->adjacentVerts();
			F(i, 0) = v[0]->id;
			F(i, 1) = v[1]->id;
			F(i, 2) = v[2]->id;
		}
		return F;
	}
	#endif

	// --------------------------------------------------------------------

	// edge length
	double length(const pte& edge) const {
		if (refPos == nullptr) { throw std::runtime_error("positions are not valid"); }

		auto i = edge->pair->vert->id, j = edge->vert->id;
		if constexpr (gp::IsEigen<Ptype>) {
			return (refPos->row(i) - refPos->row(j)).norm();
		} else if constexpr (gp::IsGlm<decltype(refPos[0][0])>) {
			#ifdef USE_GLM
			return glm::l2Norm(refPos[0][i] - refPos[0][j]);
			#endif
		} else {
			double s0 = refPos[0][i][0], s1 = refPos[0][i][1], s2 = refPos[0][i][2],
				t0 = refPos[0][j][0], t1 = refPos[0][j][1], t2 = refPos[0][j][2];
			return sqrt(pow(t0 - s0, 2) + pow(t1 - s1, 2) + pow(t2 - s2, 2));
		}
	}

	template <typename Ret>
	Ret normal(const ptv& vert) const {
		Ret n{ 0.0, 0.0, 0.0 };

		for (const auto& face : vert->adjacentFaces()) {
			if constexpr (gp::IsVec<Ret>) {
				auto norm = normal<Ret>(face);
				n[0] += norm[0];
				n[1] += norm[1];
				n[2] += norm[2];
			} else {
				n += normal<Ret>(face);
			}
		}

		return n;
	}

	template <typename Ret>
	Ret normal(const ptf& face) const {
		auto i = face->edge->vert->id;
		auto j = face->edge->next->vert->id;
		auto k = face->edge->next->next->vert->id;

		if constexpr (gp::IsEigen<Ptype>) {
			return gp::util::cross3<Ret>(refPos->row(j) - refPos->row(i), refPos->row(k) - refPos->row(i));
		} else if constexpr (gp::IsGlm<decltype(refPos[0][0])>) {
			return gp::util::cross3<Ret>(refPos[0][j] - refPos[0][i], refPos[0][k] - refPos[0][i]);
		} else {
			return gp::util::cross3<Ret>(refPos[0][i], refPos[0][j], refPos[0][k]);
		}
	}

	template <typename Ret>
	Ret center(const ptf& face) const {

		Ret c;
		if constexpr (gp::IsVec<Ret>) c.resize(3);
		for (size_t i = 0; i < 3; i++) c[i] = 0;

		auto verts = face->adjacentVerts();
		for (const auto& v : verts) {
			if constexpr (gp::IsEigen<Ptype>) {
				c[0] += refPos[0](v->id, 0);
				c[1] += refPos[0](v->id, 1);
				c[2] += refPos[0](v->id, 2);
			} else {
				c[0] += refPos[0][v->id][0];
				c[1] += refPos[0][v->id][1];
				c[2] += refPos[0][v->id][2];
			}
		}

		for (size_t i = 0; i < 3; i++) c[i] /= verts.size();

		return c;
	}

	double area(const ptf& face) const {
		if constexpr (gp::IsEigen<Ptype>) {
			#ifdef USE_EIGEN
			return 0.5 * normal<Eigen::Vector3d>(face).norm();
			#else
			throw std::runtime_error("Eigen not found");
			#endif
		} else if constexpr (gp::IsGlm<decltype(refPos[0][0])>) {
			#ifdef USE_GLM
			return 0.5 * glm::l2Norm(normal<decltype(refPos[0][0])>(face));
			#else
			throw std::runtime_error("glm not found");
			#endif
		} else {
			auto n = normal<std::vector<double>>(face);
			return 0.5 * sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
		}
	}

	double dot(const pte& e0, const pte& e1) const {
		auto i = e0->vert->id, j = e0->pair->vert->id;
		auto k = e1->vert->id, r = e1->pair->vert->id;
		if constexpr (gp::IsEigen<Ptype>) {
			return (refPos->row(i) - refPos->row(j)).dot(refPos->row(k) - refPos->row(r));
		} else if constexpr (gp::IsGlm<Ptype>) {
			#ifdef USE_GLM
			return glm::dot(refPos[0][i] - refPos[0][j], refPos[0][k] - refPos[0][r]);
			#else
			throw std::runtime_error("glm not found");
			#endif
		} else {
			return (refPos[0][i][0] - refPos[0][j][0]) * (refPos[0][k][0] - refPos[0][r][0]) +
				(refPos[0][i][1] - refPos[0][j][1]) * (refPos[0][k][1] - refPos[0][r][1]) +
				(refPos[0][i][2] - refPos[0][j][2]) * (refPos[0][k][2] - refPos[0][r][2]);
		}
	}

	double cot(const pte& edge) const {
		auto i = edge->pair->vert->id;
		auto j = edge->vert->id;
		auto k = edge->next->vert->id;
		if constexpr (gp::IsEigen<Ptype>) {
			return gp::util::cot(refPos->row(k), refPos->row(i), refPos->row(j));
		} else {
			return gp::util::cot(refPos[0][k], refPos[0][i], refPos[0][j]);
		}
	}

	double avgEdgeLength() const {
		double s = 0.0;
		for (const auto& emap : edges) {
			s += length(emap.second);
		}
		s /= (double)edges.size();
		return s;
	}

	double surfaceArea() const {
		double s = 0.0;
		for (const auto& face : faces) {
			s += area(face);
		}
		return s;
	}

};
}