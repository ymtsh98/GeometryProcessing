#pragma once
#include <kika/kikautil.hpp>
#include <kika/halfedge_mesh.hpp>
#include <kika/eigutil.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace kika::heds {

class SurfaceMesh : public HalfedgeMesh {
public:
  Eigen::MatrixX3d pos;

  SurfaceMesh() : HalfedgeMesh(), pos() {}

	/// surface mesh
	// @param pos	: positions (Eigen::MatrixX3d)
	// @param faceAttrib	: faces list (2d) or index list (1d)
	template<typename T>
	SurfaceMesh(const Eigen::MatrixX3d& pos, const T& faceAttrib) { 
		build(pos, faceAttrib);
	}

	/// surface mesh
	// @param pos	: positions (convert to Eigen::MatrixX3d)
	// @param faceAttrib	: faces list (2d) or index list (1d)
	template<typename V, typename T>
	SurfaceMesh(const std::vector<V>& vpos, const T& faceAttrib) {
		this->pos = eig::vec2eig<3>(vpos);
		build(this->pos, faceAttrib);
	}

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

	// resolve non-manifold condition vertices
	// check all vertex and edges
	void cleanVertices() override final {
		uint32_t n = (uint32_t)getNumVerts();
		pos.conservativeResize(n, 3);

		// find condition: edge has more than 3 faces (already checked as vertex index), unreference vertex

		bool existUnref = false;
		for (uint32_t i = 0; i < n; i++) {
			if (verts[i]->id != i) {
				pos.row(i) = pos.row(verts[i]->id);
				verts[i]->id = i;
			}

			existUnref |= verts[i]->isUnrefer();
		}

		if (existUnref) pos = pos(removeUnrefVerts(), Eigen::all);

		// find condition: an one-ring neighborhood is not a homeomorphism of the disk

		std::vector<bool> earrived(2 * getNumEdges(), false);
		std::vector<bool> varrived(getNumVerts(), false);
		std::vector<pte> duplicate;

		for (const auto& edge : edges) {
			if (earrived[edge->id]) continue;

			auto e = edge, e0 = e;
			do {
				earrived[e->id] = true;
				e = e->next->pair;
			} while (e != e0 && e->face != nullptr);

			if (!edge->isBoundary()) {
				if (e->face != nullptr) varrived[edge->vert->id] = true;
				continue;
			}

			if (varrived[edge->vert->id]) duplicate.emplace_back(edge);
			varrived[edge->vert->id] = true;
		}

		if (duplicate.size() == 0) return;

		pos.conservativeResize(pos.rows() + duplicate.size(), 3);

		for (auto& edge : duplicate) {
			pos.row(verts.size()) = pos.row(edge->vert->id);
			verts.emplace_back(make<Vertex>((uint32_t)verts.size()));
			verts.back()->edge = edge;
			edge->vert = verts.back();

			auto e = edge, e0 = e;
			do {
				e->vert = verts.back();
			} while (e != e0 && e->face != nullptr);
			if (e->face == nullptr) e->vert = verts.back();
		}
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
		#pragma omp parallel for
		for (int i = 0; i < (int)edges.size(); i++) {
			lengths[i] = length(edges[i]);
		}
		return lengths;
	}

	// average edge length
	double avgEdgeLength() const {
		return edgeLengths().mean();
	}

	// maximum edge length
	double maxEdgeLength() const {
		return edgeLengths().maxCoeff();
	}

	// maximum length of face diagnoal
	double maxDiagLength() const {
		double diag = 0.;
		for (size_t i = 0; i < getNumFaces(); i++) {
			auto&& vs = faces[i]->circuitVerts();
			Eigen::MatrixX3d p = pos(vs, Eigen::all);
			for (size_t j = 0; j < vs.size(); j++) {
				for (size_t k = j + 1; k < vs.size(); k++) {
					diag = std::max(diag, (p.row(j) - p.row(k)).norm());
				}
			}
		}

		return diag;
	}

	// surface area
	double surfaceArea() const {
		double s = 0.0;
		#pragma omp parallel for reduction(+:s)
		for (int i = 0; i < (int)getNumFaces(); i++) {
			s += area(faces[i]);
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

	bool isDelaunayMesh(const double eps = 1e-8) const {
		for (const auto& edge : edges) {
			if (edge->isBoundary() || edge->isK4<true, false>() || edge->isTet()) continue;

			double alpha = angle(edge->next->pair, edge->next->next);
			double beta = angle(edge->pair->next->pair, edge->pair->next->next);

			if ((alpha + beta) > std::numbers::pi + eps) return false;
		}
		return true;
	}

	// do edge flip while mesh has un-delaunay triangles (without boundary)
	void delaunayEdgeFlip(const double eps = 1e-8) {
		std::vector<bool> isDelaunayEdge(getNumEdges(), false);

		bool flip = true;
		while (flip) {
			flip = false;
			for (auto& edge : edges) {
				auto i = edge->uid();
				if (isDelaunayEdge[i]) continue;
				if (edge->isBoundary() || edge->isK4<true, false>() || edge->isTet()) continue;
			
				double alpha = angle(edge->next->pair, edge->next->next);
				double beta = angle(edge->pair->next->pair, edge->pair->next->next);
			
				if ((alpha + beta) > std::numbers::pi + eps) {
					edge->flip();
					flip = true;
					isDelaunayEdge[edge->next->uid()] = false;
					isDelaunayEdge[edge->next->next->uid()] = false;
					isDelaunayEdge[edge->pair->next->uid()] = false;
					isDelaunayEdge[edge->pair->next->next->uid()] = false;
				} 
				isDelaunayEdge[i] = true;
			}
		}
	}

	// get delaynay mesh
	SurfaceMesh delaunayMesh(const double eps = 1e-6) const {
		auto mesh = triangleMesh();
		mesh.delaunayEdgeFlip(eps);
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
		SurfaceMesh copy = *this;
		copy.triangulate();
		return copy;
	}

	// return closed mesh (hole filling)
	SurfaceMesh closedMesh() const {
		SurfaceMesh copy = *this;
		copy.fileHoles();
		return copy;
	}

	/// @return vertex-map : original mesh vertex index -> masked mesh vertex index
	/// vmap[100] -> 0, vmap[2000] -> 1
	template <typename T = uint32_t>
	std::tuple<SurfaceMesh, std::map<uint32_t, uint32_t>, std::vector<uint32_t>> fmask(const std::vector<T>& maskFacesIndex) const {
		std::vector<uint32_t> maskVerts;
		std::map<uint32_t, uint32_t> vertexMap;
		maskVerts.reserve(getNumVerts());

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

		return { SurfaceMesh(pos(maskVerts, Eigen::all), maskFaces), vertexMap, maskVerts };
	}

	/// @tparam AllOf : mask faces whitch all of vertex in maskVertsIndex
	/// @return vertex-map : original mesh vertex index -> masked mesh vertex index
	/// vmap[100] -> 0, vmap[2000] -> 1
	template <bool AllOf = true, typename T = uint32_t>
	std::tuple<SurfaceMesh, std::map<uint32_t, uint32_t>, std::vector<uint32_t>> vmask(const std::vector<T>& maskVertsIndex) const {
		std::vector<uint32_t> maskVerts;
		std::map<uint32_t, uint32_t> vertexMap;
		maskVerts.reserve(getNumVerts());

		std::vector<bool> checked(getNumVerts(), false);
		for (const auto& i : maskVertsIndex) checked[i] = true;

		std::vector<std::vector<uint32_t>> maskFaces;
		maskFaces.reserve(maskVertsIndex.size() * 2);

		uint32_t counter = 0;
		if constexpr (AllOf) {
			for (const auto& face : faces) {
				auto&& vs = face->circuitVerts();
				if (std::all_of(vs.begin(), vs.end(), [&checked](uint32_t v) { return checked[v]; })) {
					maskFaces.emplace_back(vs);

					for (auto& v : maskFaces.back()) {
						if (!vertexMap.contains(v)) {
							maskVerts.emplace_back(v);
							vertexMap[v] = counter++;
						}
						v = vertexMap[v];
					}
				}
			}
		} else {
			for (const auto& face : faces) {
				auto&& vs = face->circuitVerts();
				if (std::any_of(vs.begin(), vs.end(), [&checked](uint32_t v) { return checked[v]; })) {
					maskFaces.emplace_back(vs);

					for (auto& v : maskFaces.back()) {
						if (!vertexMap.contains(v)) {
							maskVerts.emplace_back(v);
							vertexMap[v] = counter++;
						}
						v = vertexMap[v];
					}
				}
			}
		}

		return { SurfaceMesh(pos(maskVerts, Eigen::all), maskFaces), vertexMap, maskVerts };
	}

	inline double intrinsicAngle(const size_t v, Eigen::Vector3d ref) {
		auto normal = this->normal(verts[v]);
		ref = kika::util::project2plane(ref, normal);

		auto&& edges = verts[v]->incidentEdges<kika::heds::Ptr, kika::heds::CCW>();

		size_t minIndex = 0;
		double minAngle = 100.;

		for (size_t i = 0; i < edges.size(); i++) {
			Eigen::Vector3d e = - this->edgevec(edges[i]);
			double angle = kika::util::angle(e, ref, normal);
			if (fabs(angle) < minAngle) {
				minAngle = fabs(angle);
				minIndex = i;
			}
		}

		double theta = 0.0;
		Eigen::Vector3d e0 = - this->edgevec(edges[0]);
		for (size_t i = 1; i < minIndex; i++) {
			Eigen::Vector3d e = - this->edgevec(edges[i]);
			double angle = kika::util::angle(e0, e, normal);
			theta += angle;
			e0 = e;
		}

		return theta + kika::util::angle(e0, ref, normal);
	}
};

}