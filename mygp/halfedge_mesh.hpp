#pragma once
#include <memory>
#include <unordered_map>
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

#include <mygp/geoutil.hpp>

namespace heds {
	class Vertex;
	class Halfedge;
	class Face;

	using ptv = std::shared_ptr<Vertex>;
	using pte = std::shared_ptr<Halfedge>;
	using ptf = std::shared_ptr<Face>;

	using eKey = std::pair<uint32_t, uint32_t>;
	inline eKey keyPair(const eKey& key) { return eKey(key.second, key.first); }
	struct eKeyHash {
		size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
			return std::hash<uint32_t>()(p.first) ^ std::hash<uint32_t>()(p.second);
		}
	};

	using ptVerts = std::vector<ptv>;
	using ptEdges = std::unordered_map<eKey, pte, eKeyHash>;
	using ptFaces = std::vector<ptf>;

	class Vertex {
	public:
		pte edge;

		uint32_t id;

		Vertex() : edge(nullptr), id(0) {}

		bool operator==(const Vertex& other) {
			return (id == other.id);
		}

	public:
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

		Halfedge() : next(nullptr), pair(nullptr), face(nullptr), vert(nullptr) {}

		// get adjacent faces
		std::pair<uint32_t, uint32_t> adjacentFaces() const;

	private:
		void contractSwitch();

	public:

		// contract this edge (triangle only)
		void contract(ptv& vert);

		bool isContractible() const;
	};

	class Face {
	public:
		pte edge;
		uint32_t id;

		Face() : edge(nullptr), id(0) {}

		// get vertices (ordered counter-clockwise)
		ptVerts adjacentVerts() const;

		bool isInvalid() const;
	};

	std::pair<uint32_t, uint32_t> Halfedge::adjacentFaces() const {
		uint32_t i = face != nullptr ? face->id : std::numeric_limits<uint32_t>::max();
		uint32_t j = pair->face != nullptr ? pair->face->id : std::numeric_limits<uint32_t>::max();
		return std::pair<uint32_t, uint32_t>(i, j);
	}

	ptVerts Vertex::oneringNeighbs() const {
		ptVerts neighbs;
		neighbs.reserve(6);
		pte e = this->edge;
		
		do {
			e = e->next;
			neighbs.emplace_back(e->vert);
			e = e->pair;
		} while (e->face != nullptr && e != this->edge);
		
		return neighbs;
	}

	std::vector<pte> Vertex::incidentEdges() const {
		std::vector<pte> edges;
		pte e = this->edge;
		
		do {
			edges.emplace_back(e);
			e = e->next->pair;
		} while (e->face != nullptr && e != this->edge);
		if (e->face == nullptr) edges.emplace_back(e);

		return edges;
	}

	ptFaces Vertex::adjacentFaces() const {
		ptFaces faces;
		faces.reserve(6);
		auto e = this->edge;
		
		do {
			faces.emplace_back(e->face);
			e = e->next->pair;
		} while (e->face != nullptr && this->edge != e);
		
		return faces;
	}

	bool Vertex::isBoundary() const {
		return edge->pair->face == nullptr;
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
		auto es0 = this->vert->incidentEdges();
		auto es1 = this->pair->vert->incidentEdges();

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
		if (this->face != nullptr && this->pair->face != nullptr && 
				this->vert->isBoundary() && this->pair->vert->isBoundary()) return false;

		int count = (face == nullptr || pair->face == nullptr);
		std::unordered_map<ptv, int> map;
		for (const auto& v : vert->oneringNeighbs()) count += map[v]++;
		for (const auto& v : pair->vert->oneringNeighbs()) count += map[v]++;
		return count < 3;
	}

	ptVerts Face::adjacentVerts() const {
		ptVerts verts;
		verts.reserve(3);
		pte e = this->edge;

		do {
			verts.emplace_back(e->vert);
			e = e->next;
		} while (e != this->edge);

		return verts;
	}

	bool Face::isInvalid() const {
		return this->edge != this->edge->pair->pair ||
			this->edge->next != this->edge->next->pair->pair ||
			this->edge->next->next != this->edge->next->next->pair->pair;
	}
}

namespace heds {
	// halfedge mesh class
	// requires gp::Is2DVec<Ptype> || gp::IsEigen<Ptype>
	template <typename Ptype = std::vector<std::vector<double>>>
	class Mesh {

		Ptype* refPos;

		template <typename T>
		void initVertices(T& positions) {
			if constexpr (gp::IsEigen<T>) verts.resize(positions.rows());
			else if constexpr (gp::IsScalar<T>) verts.resize(positions);
			else verts.resize(positions.size());

			for (uint32_t i = 0; i < verts.size(); i++) {
				verts[i] = std::make_shared<Vertex>();
				verts[i]->id = i;
			}
		}

		template <typename T>
		void initFaces(const T& faceIndices) {
			uint32_t n;
			if constexpr (gp::IsEigen<T>) n = (uint32_t)faceIndices.rows();
			else if constexpr (gp::Is2DVec<T>) n = (uint32_t)faceIndices.size();
			else n = (uint32_t)(faceIndices.size() / 3);

			faces.reserve(n);
			for (uint32_t i = 0; i < n; i++) {
				std::vector<eKey> feKey;
				if constexpr (gp::IsEigen<T>) {
					feKey.reserve(faceIndices.cols());
					for (size_t j = 0, k = (size_t)faceIndices.cols(); j < k; j++) 
						feKey.emplace_back(faceIndices(i, j), faceIndices(i, (j+1)%k));
				} else if constexpr (gp::Is2DVec<T>) {
					feKey.reserve(faceIndices[i].size());
					for (size_t j = 0, k = faceIndices[i].size(); j < k; j++) 
						feKey.emplace_back(faceIndices[i][j], faceIndices[i][(j+1)%k]);
				} else {
					feKey.reserve(3);
					feKey.emplace_back(faceIndices[i * 3 + 0], faceIndices[i * 3 + 1]);
					feKey.emplace_back(faceIndices[i * 3 + 1], faceIndices[i * 3 + 2]);
					feKey.emplace_back(faceIndices[i * 3 + 2], faceIndices[i * 3 + 0]);
				}

				faces.emplace_back(std::make_shared<Face>());

				std::vector<bool> flip(feKey.size(), false);
				for (size_t j = 0; j < feKey.size(); j++) {
					auto first = feKey[j].first, second = feKey[j].second;
					std::shared_ptr<Halfedge> edge;

					if (edges.contains(keyPair(feKey[j]))) {
						feKey[j] = keyPair(feKey[j]);
						edge = edges.at(feKey[j])->pair;
						flip[j] = true;
					} else {
						edge = std::make_shared<Halfedge>();
						edge->pair = std::make_shared<Halfedge>();
						edge->pair->pair = edge;
						edges.emplace(feKey[j], edge);
					}

					edge->face = faces[i];
					edge->vert = verts[second];
					edge->pair->vert = verts[first];
					verts[second]->edge = edge;
					faces[i]->edge = edge;
				}

				for (size_t j = 0, k = feKey.size(); j < k; j++) {
					auto& edge =  flip[j] ? edges.at(feKey[j])->pair : edges.at(feKey[j]);
					auto& nKey = (j + 1) < k ? feKey[j+1] : feKey[0];
					edge->next = flip[(j + 1) % k] ? edges.at(nKey)->pair : edges.at(nKey);
				}

				faces[i]->id = i;
			}
		}

		void initBoundary() {
			for (const auto& emap : edges) {
				if (emap.second->pair->face == nullptr) 
					verts[emap.first.second]->edge = emap.second;
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

		size_t getNumVertices() const {
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

		// bind positions
		void bind(Ptype& bindPos) {
			refPos = &bindPos;
		}

		// builds halfedge data structure
		// @param positions	: vertex matrix (2d)
		// @param faceIndices	: face matrix (2d) or index list (1d)
		template<typename T1>
		void build(Ptype& positions, const T1& faceIndices) {
			bind(positions);
			initVertices(positions);
			initFaces(faceIndices);
			initBoundary();
		}

		// builds halfedge data structure
		// @param n		: number of vertices (scalar)
		// @param faceIndices	: face matrix (2d) or index list (1d)
		template<gp::IsScalar T1, typename T2>
		void build(const T1& n, const T2& faceIndices) {
			initVertices(n);
			initFaces(faceIndices);
			initBoundary();
		}

		void rebuildEdges() {
			edges.clear();

			for (const auto& face : this->faces) {
				auto e0 = face->edge;
				auto e = face->edge;
				do {
					eKey key(e->vert->id, e->pair->vert->id);
					if (!edges.contains(keyPair(key))) edges[key] = e;
					e = e->next;
				} while (e0 != e);
			}
		}

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
				return gp::cross3<Ret>(refPos->row(j) - refPos->row(i), refPos->row(k) - refPos->row(i));
			} else if constexpr (gp::IsGlm<decltype(refPos[0][0])>) {
				return gp::cross3<Ret>(refPos[0][j] - refPos[0][i], refPos[0][k] - refPos[0][i]);
			} else {
				return gp::cross3<Ret>(refPos[0][i], refPos[0][j], refPos[0][k]);
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
				return gp::cot(refPos->row(k), refPos->row(i), refPos->row(j));
			} else {
				return gp::cot(refPos[0][k], refPos[0][i], refPos[0][j]);
			}
		}

		double meanEdgeLength() const {
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

		Ptype copyPosition() {
			return refPos[0];
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
		heds::ptv appendPosition(const T& pos) {
			heds::ptv p = std::make_shared<Vertex>();

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
			std::vector<std::vector<uint32_t>> adlist(this->getNumVertices());

			for (size_t i = 0; i < this->getNumVertices(); i++) {
				for (const auto& v: this->verts[i]->oneringNeighbs()) adlist[i].emplace_back(v->id);
			}
			return adlist;
		}

	};
}