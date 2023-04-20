#pragma once

#include <kika/halfedge_mesh.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace kika::heds {

class SurfaceMesh : public HalfedgeMesh {
public:
  Eigen::MatrixX3d pos;

  SurfaceMesh() : pos(), HalfedgeMesh() {}

	/// surface mesh
	// @param pos	: positions (Eigen::MatrixX3d)
	// @param faceAttrib	: faces list (2d) or index list (1d)
	template<typename T>
	SurfaceMesh(const Eigen::MatrixX3d& pos, const T& faceAttrib) : pos(pos), HalfedgeMesh(pos.rows(), faceAttrib) { }

	SurfaceMesh(const SurfaceMesh& mesh) {
		build(mesh.pos, mesh.faceList());
	}

	SurfaceMesh& operator=(const SurfaceMesh& mesh) {
		reset(true);
		build(mesh.pos, mesh.faceList());
		return *this;
	}

	/// surface mesh
	// @param pos	: positions (Eigen::MatrixX3d)
	// @param faceAttrib	: faces list (2d) or index list (1d)
	template<typename T>
	void build(const Eigen::MatrixX3d& pos, const T& faceAttrib) {
		this->pos = pos;
		HalfedgeMesh::build(pos.rows(), faceAttrib);
	}


	// --------------- geometry ---------------------

	Eigen::Vector3d edgevec(const pte& edge) const {
		return pos.row(edge->vert->id) - pos.row(edge->pair->vert->id);
	}

	Eigen::Vector3d center(const pte& edge) const {
		return 0.5 * (pos.row(edge->vert->id) + pos.row(edge->pair->vert->id));
	}

	Eigen::Vector3d center(const ptf& face) const {
		return pos(face->circuitVerts(), Eigen::all).colwise().mean();
	}

	Eigen::Vector3d normal(const ptf& face) const {
		Eigen::Vector3d n = Eigen::Vector3d::Zero();
		auto&& vidx = face->circuitVerts();
		for (size_t i = 0, k = vidx.size(); i < k; i++) {
			n += pos.row(vidx[i]).cross(pos.row(vidx[(i+1)%k]));
		}
		return n;
	}

	Eigen::Vector3d normal(const ptv& vert) const {
		Eigen::Vector3d n = Eigen::Vector3d::Zero();

		for (const auto& face : vert->adjacentFaces<Ptr>()) {
			n += normal(face);
		}

		return n.normalized();
	}

	// edge length
	double length(const pte& edge) const {
		return (pos.row(edge->pair->vert->id) - pos.row(edge->vert->id)).norm();
	}

	// face area
	double area(const ptf& face) const {
		return 0.5 * normal(face).norm();
	}

	// cotangent weight
	double cot(const pte& edge) const {
		auto i = edge->pair->vert->id;
		auto j = edge->vert->id;
		auto k = edge->next->vert->id;
		auto e1 = (pos.row(i) - pos.row(k)).eval();
		auto e2 = (pos.row(j) - pos.row(k)).eval();
		return e1.dot(e2)/ e1.cross(e2).norm();
	}

	// dot product between e0 and e1
	double dot(const pte& e0, const pte& e1) const {
		return edgevec(e0).dot(edgevec(e1));
	}

	// unsigned angle between e0 and e1
	double angle(const pte& e0, const pte& e1) const {
		return util::angle(edgevec(e0), edgevec(e1));
	}

	// ----------------- element operation ---------------------
	
private:
	struct PosSwapper {
		void operator()(const uint32_t i, Eigen::MatrixX3d& pos) {
			pos.row(i) = pos.bottomRows(1);
			pos.conservativeResize(pos.rows() - 1, pos.cols());
		}
	};

public:
	template <typename SwapObj = Eigen::MatrixX3d, typename SwapFunc = PosSwapper>
	bool remove(ptv vert, SwapObj* swapOrder = nullptr) {
		if constexpr (std::is_same_v<SwapFunc, PosSwapper>)
			return HalfedgeMesh::remove<Eigen::MatrixX3d, PosSwapper>(vert, &pos);
		else 
			return HalfedgeMesh::remove<SwapObj, SwapFunc>(vert, swapOrder);
	}

	template <typename SwapObj = Eigen::MatrixX3d, typename SwapFunc = PosSwapper>
	bool remove(pte edge, SwapObj* swapOrder = nullptr) {
		if constexpr (std::is_same_v<SwapFunc, PosSwapper>)
			return HalfedgeMesh::remove<Eigen::MatrixX3d, PosSwapper>(edge, &pos);
		else
			return HalfedgeMesh::remove<SwapObj, SwapFunc>(edge, swapOrder);
	}

	template <typename SwapObj = Eigen::MatrixX3d, typename SwapFunc = PosSwapper>
	bool remove(ptf face, SwapObj* swapOrder = nullptr) {
		if constexpr (std::is_same_v<SwapFunc, PosSwapper>)
			return HalfedgeMesh::remove<Eigen::MatrixX3d, PosSwapper>(face, &pos);
		else
			return HalfedgeMesh::remove<SwapObj, SwapFunc>(face, swapOrder);
	}

private:
	struct PosInserter {
		void operator()(const uint32_t i, Eigen::MatrixX3d& pos) {
			pos.conservativeResize(pos.rows() + 1, 3);
			pos.bottomRows(1) = pos.row(i);
		}
	};

public:
	// edge split
	// @return inserted vertex index (default std::numerlic_limits<uint32_t>::max())
	template <typename InstObj = Eigen::MatrixX3d, typename InstFunc = PosInserter>
	uint64_t split(pte edge, InstObj* instOrder = nullptr) {
		if constexpr (std::is_same_v<InstFunc, PosInserter>)
			return HalfedgeMesh::split<Eigen::MatrixX3d, PosInserter>(edge, &pos);
		else
			return HalfedgeMesh::split<InstObj, InstFunc>(edge, instOrder);
	}

	// ----------------- mesh operation ---------------------
	
	Eigen::VectorXd edgeLengths() const {
		Eigen::VectorXd lengths(getNumEdges());
		for (size_t i = 0; i < edges.size(); i++) {
			lengths[i] = length(edges[i]);
		}
		return lengths;
	}

	// average edge length
	double avgEdgeLength() const {
		return edgeLengths().mean();
	}


	// surface area
	double surfaceArea() const {
		double s = 0.0;
		for (const auto& face : faces) {
			s += area(face);
		}
		return s;
	}
	
	// area weighted center
	Eigen::Vector3d weightedCenter() const {
		double sa = 0.0;
		Eigen::Vector3d center(0., 0., 0.);
		for (const auto& face : faces) {
			double a = area(face);
			sa += a;
			center += a * pos(face->circuitVerts(), Eigen::all).colwise().mean();
		}
		center /= sa;
		return center;
	}

	Eigen::MatrixX3d faceNormals() const {
		Eigen::MatrixX3d normals = Eigen::MatrixX3d::Zero(getNumFaces(), 3);
		for (size_t i = 0; i < getNumFaces(); i++) {
			normals.row(i) = normal(faces[i]);
		}
		return normals;
	}

	Eigen::MatrixX3d vertexNormals() const {
		Eigen::MatrixX3d normals = Eigen::MatrixX3d::Zero(getNumVerts(), 3);
		for (size_t i = 0; i < getNumFaces(); i++) {
			auto n = normal(faces[i]);
			for (const auto& v : faces[i]->circuitVerts()) {
				normals.row(v) += n.transpose();
			}
		}
		normals.rowwise().normalize();
		return normals;
	}

	// get dual mesh
	// preserve prim->dual->prim(pdp) mesh boundaries : 
	// for (auto& face : pdp.faces) {
	// 	if (face->isBoundary()) prim.remove(face);
	// }
	SurfaceMesh dualMesh() const {
		auto dfaces = dualFaceList();
		Eigen::MatrixX3d c(getNumFaces() + getNumBoundaryEdges(), 3);

		size_t i = 0;
		for (; i < getNumFaces(); i++) {
			c.row(i) = center(faces[i]);
		}

		for (const auto& edge : edges) {
			if (edge->isBoundary()) c.row(i++) = center(edge);
		}
		return SurfaceMesh(c, dfaces);
	}

	bool isDelaunayMesh() const {
		for (const auto& edge : edges) {
			if (edge->isBoundary() || edge->isK4<true, false>()) continue;

			double alpha = angle(edge->next->pair, edge->next->next);
			double beta = angle(edge->pair->next->pair, edge->pair->next->next);

			if ((alpha + beta) > std::numbers::pi) return false;
		}
		return true;
	}

	// do edge flip while mesh has un-delaunay triangles (without boundary)
	void delaunayEdgeFlip() {
		bool flip = true;
		while (flip) {
			static int counter = 0;
			flip = false;
			for (auto& edge : edges) {
				if (edge->isBoundary() || edge->isK4<true, false>()) continue;

				double alpha = angle(edge->next->pair, edge->next->next);
				double beta = angle(edge->pair->next->pair, edge->pair->next->next);

				if ((alpha + beta) > std::numbers::pi) {
					edge->flip();
					flip = true;
				}
			}
		}
	}

	// get delaynay mesh
	SurfaceMesh delaunayMesh() const {
		auto mesh = triangleMesh();
		mesh.delaunayEdgeFlip();
		return mesh;
	}

	// normalize mesh
	void normalize(const bool normWithSurfaceArea = false, const bool areaWeightedCenter = false) {
		double sa = 0.0;
		if (areaWeightedCenter) {
			Eigen::Vector3d center(0., 0., 0.);
			for (const auto& face : faces) {
				double a = area(face);
				sa += a;
				center += a * pos(face->circuitVerts(), Eigen::all).colwise().mean();
			}
			center /= sa;
			pos.array().rowwise() -= center.transpose().array();
		} else {
			pos.array().rowwise() -= pos.colwise().mean().array();
		}
		if (normWithSurfaceArea) {
			if (sa == 0.) sa = surfaceArea();
			pos.array() *= sqrt(1. / sa);
		} else {
			pos.array() /= pos.maxCoeff();
		}
	}

	SurfaceMesh triangleMesh() const {
		return SurfaceMesh(pos, util::faces2indices(faceList()));
	}
};

}