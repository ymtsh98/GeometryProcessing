#pragma once
#include<iostream>
#include <array>
#include <utility>
#include <memory>
#include <map>
#include <vector>

#if __has_include("glm/glm.hpp")
#include <glm/glm.hpp>
#define GLM_INCLUDE
#endif
#if __has_include("Eigen/Eigen")
#include "Eigen/Core"
#define EIGEN_INCLUDE
#endif

namespace heds {
	class Mesh;

	class Face;
	class Halfedge;
	class Vertex;

	using ptf = std::shared_ptr<Face>;
	using pte = std::shared_ptr<Halfedge>;
	using ptv = std::shared_ptr<Vertex>;

	using eKey = std::pair<ptv, ptv>;
	using ptEdges = std::map<eKey, pte>;
	using ptFaces = std::vector<ptf>;
	using ptVerts = std::vector<ptv>;

	union dvec3 {
		std::array<double, 3> arr;
		#ifdef GLM_INCLUDE
		glm::dvec3 glm;
		#endif
		#ifdef EIGEN_INCLUDE
		Eigen::Vector3d eig;
		#endif

		dvec3() {}
		dvec3(double x) : arr{ x, x, x } {}
		dvec3(double x, double y, double z) : arr{ x, y, z } {}

		double& operator[](int i) {
			return arr[i];
		}
	};

	template <typename T>
	double norm(const T& a) {
		return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	}
}

namespace heds {
	class Halfedge {
	public:
		pte next;
		pte prev;
		pte pair;

		ptf face;
		ptv vert;

		Halfedge() : next(nullptr), prev(nullptr), pair(nullptr), face(nullptr), vert(nullptr) {}

		double length();
	};

	class Vertex {
	public:
		pte edge;
		dvec3 pos;

		uint32_t id;
		bool is_boundary;

		Vertex() : edge(nullptr), pos(0, 0, 0), id(0), is_boundary(false) {}
		template <typename T>
		Vertex(T p) : edge(nullptr), pos(p[0], p[1], p[2]), id(0), is_boundary(false) {}

		bool operator==(const Vertex& other) {
			return (pos.glm == other.pos.glm) && (id == other.id);
		}

		friend std::ostream& operator<<(std::ostream& os, const Vertex& v);
	};
	
	std::ostream& operator<<(std::ostream& os, const Vertex& v) {
		os << v.id << " : " << v.pos.arr[0] << " " << v.pos.arr[1] << " " << v.pos.arr[2];
		return os;
	}

	double Halfedge::length() {
		auto s = prev->vert->pos.arr;
		auto t = vert->pos.arr;
		return sqrt(pow(t[0] - s[0], 2) + pow(t[1] - s[1], 2) + pow(t[2] - s[2], 2));
	}
	class Face {
	public:
		pte edge;
		uint32_t id;

		Face() : edge(nullptr), id(0) {}

		template <typename T>
		T normal() {
			auto a = this->edge->vert->pos.arr;
			auto b = this->edge->next->vert->pos.arr;
			auto c = this->edge->next->next->vert->pos.arr;

			return {
				(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
				(b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
				(b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) };
		}

		double area() {
			#ifdef GLM_INCLUDE
			auto a = this->edge->vert->pos.glm;
			auto b = this->edge->next->vert->pos.glm;
			auto c = this->edge->next->next->vert->pos.glm;
			
			return 0.5 * glm::length(glm::cross(b - a, c - a));
			#else
			return 0.5 * norm(normal<std::array<double, 3>>());
			#endif
		}

	};

	class Mesh {
	public:
		ptEdges edges;
		ptFaces faces;
		ptVerts verts;

		Mesh() {}
		
		template<typename T1, typename T2>
		Mesh(const T1& positions, const T2& faces) {
			init(positions, faces);
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

		template <typename T1, typename T2>
		void init(const T1& positions, const T2& indices) {
			verts.resize(positions.size());
			for (size_t i = 0; i < verts.size(); i++) {
				verts[i] = std::make_shared<Vertex>();
				verts[i]->pos.arr = { positions[i][0], positions[i][1], positions[i][2] };
				verts[i]->id = i;
			}

			faces.reserve(indices.size());
			for (uint32_t fid = 0; auto & idx : indices) {
				std::vector<eKey> fe(idx.size());
				for (size_t i = 0; i < fe.size() - 1; i++) fe[i] = { verts[idx[i]], verts[idx[i + 1]] };
				fe[fe.size() - 1] = { verts[idx[fe.size() - 1]], verts[idx[0]] };

				faces.push_back(std::make_shared<Face>());

				for (auto& e : fe) {
					edges[e] = std::make_shared<Halfedge>();
					edges[e]->face = faces[fid];
					edges[e]->vert = e.second;
					e.second->edge = edges[e];
				}

				for (auto e_itr = fe.begin(); e_itr < fe.end(); e_itr++) {
					auto e = *e_itr;
					auto next_e = *(e_itr + 1 < fe.end() ? (e_itr + 1) : fe.begin());
					auto prev_e = *(e_itr > fe.begin() ? (e_itr - 1) : fe.end() - 1);
					eKey pair_e = { e.second, e.first };

					edges[e]->next = edges[next_e];
					edges[e]->prev = edges[prev_e];

					if (edges.count(pair_e) != 0) {
						edges[e]->pair = edges[pair_e];
						edges[pair_e]->pair = edges[e];
					}
				}

				faces[fid]->edge = edges[fe[0]];
				faces[fid]->id = fid;
				fid++;
			}

			initBound();
		}

		void initBound() {
			for (auto& f : faces) {
				pte& e0 = f->edge;
				pte edge = e0;

				do {
					if (edge->pair == nullptr) {
						edge->vert->is_boundary = true;
						edge->vert->edge = edge;
						edge->prev->vert->is_boundary = true;
						edge->prev->vert->edge = edge->prev;
					}

					edge = edge->next;
				} while (edge != e0);

			}
		}

		double meanEdgeLength() {
			double s = 0.0;
			for (auto& face : faces) {
				auto e1 = face->edge;
				auto e2 = face->edge->next;
				auto e3 = face->edge->next->next;

				s += e1->length() + e2->length() + e3->length();
			}

			s /= (3.0 * faces.size());
			return s;
		}

		double surfaceArea() {
			double s = 0.0;
			for (auto& face : faces) {
				s += face->area();
			}
			return s;
		}
	};

	ptVerts adjacentVerts(const ptf& face) {
		pte& e0 = face->edge;
		ptVerts verts = { e0->vert };
		pte edge = e0->next;

		while (edge != e0) {
			verts.push_back(edge->vert);
			edge = edge->next;
		}

		return verts;
	}

	ptVerts oneringNeighbsCW(pte edge) {
		ptVerts neighbs;

		while (edge->pair != nullptr) {
			neighbs.push_back(edge->pair->vert);
			edge = edge->pair->prev;
		}

		neighbs.push_back(edge->prev->vert);
		return neighbs;
	}

	ptVerts oneringNeighbs(const ptv& vertex) {
		pte& e0 = vertex->edge;
		pte edge = e0->next;

		ptVerts neighbs;

		if (e0->pair != nullptr) neighbs.push_back(e0->pair->vert);
		else neighbs.push_back(e0->prev->vert);

		while (edge->pair != e0) {
			neighbs.push_back(edge->vert);
			if (edge->pair == nullptr) break;
			edge = edge->pair->next;
		}

		if (edge->pair == nullptr && e0->pair != nullptr) {
			auto neighbsCW = oneringNeighbsCW(e0);
			neighbs.insert(neighbs.end(), neighbsCW.begin(), neighbsCW.end());
		}

		return neighbs;
	}

	ptFaces adjacentFacesCW(pte edge) {
		ptFaces faces;

		while (edge->next->pair != nullptr) {
			edge = edge->next->pair;
			faces.push_back(edge->face);
		}

		return faces;
	}

	ptFaces adjacentFaces(const ptv& vertex) {
		pte& e0 = vertex->edge;
		ptFaces faces = { e0->face };

		pte edge = e0;
		if (e0->pair != nullptr) edge = e0->pair->prev;

		while (edge != e0) {
			faces.push_back(edge->face);

			if (edge->pair == nullptr) break;
			edge = edge->pair->prev;
		}

		if (edge->pair == nullptr) {
			auto facesCW = adjacentFacesCW(e0);
			faces.insert(faces.end(), facesCW.begin(), facesCW.end());
		}

		return faces;
	}
}