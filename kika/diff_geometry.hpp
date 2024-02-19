#pragma once
#include <kika/surface_mesh.hpp>
#include <kika/eigutil.hpp>

namespace kika::diff {

// 
// FwDeltaOperator delOp;
// delOp.compute(mesh);
// 
// auto gradf = delOp(source);
// auto divf  = delOp * gradf;
// auto curlf = delOp ^ gradf;
// divf  = delOp * curlf == [0];
class FwDeltaOperator {
	using SparseMatrices3d = std::array<Eigen::SparseMatrix<double>, 3>;

	SparseMatrices3d del; // |F|x|V|x3 delta operator
	Eigen::MatrixX3d fns; // face unit normal
	// SparseMatrixces3d fes; // |F|x|V| inner edges, del = fns.diag.3d x fes
	// - fes(scalar_func) -> rot 90 deg of gradient

	Eigen::VectorXd fareas;

public:

	FwDeltaOperator() {}

	void reset() {
		for (int i = 0; i < 3; i++) del[i].resize(0, 0);
		fns.resize(0, 0);
		fareas.resize(0);
	}

	Eigen::SparseMatrix<double>& operator[](int i) {
		assert(i < 3 && "FwDeltaOperator");
		return del[i];
	}

	void compute(const kika::heds::SurfaceMesh& mesh) {
		size_t nv = mesh.getNumVerts(), nf = mesh.getNumFaces();

		fns.resize(nf, 3);
		fareas = Eigen::VectorXd::Zero(nf);

		std::array<std::vector<Eigen::Triplet<double>>, 3> tripls;
		for (int i = 0; i < 3; i++) tripls[i].reserve(nf * 4);

		for (size_t i = 0; i < nf; i++) {
			auto normal = mesh.normal(mesh.faces[i]);
			fareas[i] = normal.norm();
			normal.array() /= fareas[i]; // normalize
			fns.row(i) = normal;
			normal.array() /= fareas[i]; // divided by area

			auto&& vidx = mesh.faces[i]->circuitVerts();
			for (size_t j = 0, k = vidx.size(); j < k; j++) {
				Eigen::Vector3d e = normal.cross(mesh.pos.row(vidx[(j+2)%k]) - mesh.pos.row(vidx[(j+1)%k]));
				for (int r = 0; r < 3; r++) tripls[r].emplace_back((int)i, (int)vidx[j], e(r));
			}
		}

		for (int i = 0; i < 3; i++) {
			del[i].resize(nf, nv);
			del[i].setFromTriplets(tripls[i].begin(), tripls[i].end());
		}
	}

	// gradient
	inline Eigen::MatrixX3d operator()(const Eigen::VectorXd& v) const {
		Eigen::MatrixX3d ret(del[0].rows(), 3);
		for (int i = 0; i < 3; i++)
			ret.col(i) = del[i] * v;
		return ret;
	}

	// curl
	inline Eigen::MatrixX3d operator^(const Eigen::VectorXd& v) const {
		return this->operator^(this->operator()(v));
	}

	// curl
	inline Eigen::MatrixX3d operator^(const Eigen::MatrixX3d& m) const {
		Eigen::MatrixX3d ret(del[0].rows(), 3);

		ret.col(0) = fns.col(1).asDiagonal() * m.col(2) - fns.col(2).asDiagonal() * m.col(1);
		ret.col(1) = fns.col(2).asDiagonal() * m.col(0) - fns.col(0).asDiagonal() * m.col(2);
		ret.col(2) = fns.col(0).asDiagonal() * m.col(1) - fns.col(1).asDiagonal() * m.col(0);

		return ret;
	}

	// delta * delta (divergence * gradient)
	inline Eigen::SparseMatrix<double> operator*(const FwDeltaOperator& other) const {
		return (del[0].transpose() * fareas.asDiagonal() * other.del[0]
			+ del[1].transpose() * fareas.asDiagonal() * other.del[1]
			+ del[2].transpose() * fareas.asDiagonal() * other.del[2]);
	}

	// divergence
	inline Eigen::VectorXd operator*(const Eigen::MatrixX3d& m) const {
		return (del[0].transpose() * fareas.asDiagonal() * m.col(0)
			+ del[1].transpose() * fareas.asDiagonal() * m.col(1)
			+ del[2].transpose() * fareas.asDiagonal() * m.col(2));
	}

};

class SparseMatrices3d {
	std::array<Eigen::SparseMatrix<double>, 3> mat;
public:

	void resize(const size_t rows, const size_t cols) {
		for (int i = 0; i < 3; i++) mat[i].resize(rows, cols);
	}

	Eigen::SparseMatrix<double>& operator[](int i) {
		assert(i < 3 && "SparseMatrices");
		return mat[i];
	}

	// for gradient
	Eigen::MatrixX3d operator*(const Eigen::VectorXd& v) const {
		Eigen::MatrixX3d ret(mat[0].rows(), 3);
		for (int i = 0; i < 3; i++)
			ret.col(i) = mat[i] * v;
		return ret;
	}

	// for curl
	Eigen::MatrixX3d operator^(const Eigen::MatrixX3d& m) const {
		Eigen::MatrixX3d ret(mat[0].rows(), 3);
		for (int i = 0; i < 3; i++)
			ret.col(i) = mat[(i + 1) % 3] * m.col((i + 2) % 3) - mat[(i + 2) % 3] * m.col((i + 1) % 3);
		return ret;
	}

	// for divergence
	Eigen::SparseMatrix<double> operator*(const SparseMatrices3d& other) const {
		return (mat[0] * other.mat[0] + mat[1] * other.mat[1] + mat[2] * other.mat[2]);
	}

	// for divergence
	Eigen::VectorXd operator*(const Eigen::MatrixX3d& m) const {
		return (mat[0] * m.col(0) + mat[1] * m.col(1) + mat[2] * m.col(2));
	}
};

}

namespace kika::diff {

class DiscreteLaplacian {
protected:
	const heds::SurfaceMesh* mesh;
public:

	mutable Eigen::SparseMatrix<double> lap; // vertex-wise laplace-beltrami operator (stiffness term)

	mutable SparseMatrices3d grad; // |F|x|V| gradient operator
	mutable SparseMatrices3d div;  // |V|x|F| divergence operator
	mutable SparseMatrices3d curl; // |F|x|F| curl operator

	mutable Eigen::VectorXd hstar0; // 0-form hodge star : prim vert <-> dual face
	mutable Eigen::VectorXd hstar1; // 1-form hodge star : prim edge <-> dual edge
	mutable Eigen::VectorXd hstar2; // 2-form hodge star : prim face <-> dual vert

	mutable Eigen::SparseMatrix<double> ederiv0; // 0-form exterior derivative : |E|x|V|
	mutable Eigen::SparseMatrix<double> ederiv1; // 1-form exterior derivative : |F|x|E|

	mutable Eigen::SparseMatrix<Eigen::dcomplex> clap; // vertex-wise connection laplacian

	// grad : d0, del()
	// curl : (s2) * d1, del ^
	// div  : (s0^-1) * d0.T * s1, del * 

	//// test : curl of gradient = 0 : d1 * d0 * vwise_func;
	// Eigen::MatrixX3d grad_pos = (d0 * mesh.pos);
	// Eigen::MatrixX3d curl_grad_pos = d1 * grad_pos; // [[0 0 0] ... [0 0 0]]

	DiscreteLaplacian() : mesh(nullptr) {}

	DiscreteLaplacian(const heds::SurfaceMesh& mesh) : mesh(&mesh) {}

	virtual ~DiscreteLaplacian() {}

	void build(const heds::SurfaceMesh& mesh) {
		this->mesh = &mesh;
		reset();
	}

	void reset() {
		lap.resize(0, 0);
		grad.resize(0, 0);
		div.resize(0, 0);
		curl.resize(0, 0);

		hstar0.resize(0);
		hstar1.resize(0);
		hstar2.resize(0);
		ederiv0.resize(0, 0);
		ederiv1.resize(0, 0);
	}

	// vertex-wise laplacian for triangle mesh
	virtual const Eigen::SparseMatrix<double>& laplacianOp() const {
		if (lap.size() != 0) return lap;

		auto [nv, ne, nf] = mesh->vef();

		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(9 * nf);

		for (const auto& face : mesh->faces) {
			auto& e0 = face->edge;
			auto& e1 = e0->next;
			auto& e2 = e1->next;

			double w0 = 0.5 * mesh->cot(e0);
			double w1 = 0.5 * mesh->cot(e1);
			double w2 = 0.5 * mesh->cot(e2);

			auto v0 = e0->vert->id;
			auto v1 = e1->vert->id;
			auto v2 = e2->vert->id;

			triplets.emplace_back(v0, v0, (w0 + w1));
			triplets.emplace_back(v1, v1, (w1 + w2));
			triplets.emplace_back(v2, v2, (w2 + w0));

			triplets.emplace_back(v0, v1, -w1);
			triplets.emplace_back(v1, v0, -w1);
			triplets.emplace_back(v1, v2, -w2);
			triplets.emplace_back(v2, v1, -w2);
			triplets.emplace_back(v2, v0, -w0);
			triplets.emplace_back(v0, v2, -w0);
		}

		lap.resize(nv, nv);
		lap.setFromTriplets(triplets.begin(), triplets.end());

		return lap;
	}

	// gradient operator
	virtual const SparseMatrices3d& gradientOp() const {
		if (grad[0].size() != 0) return grad;

		size_t nv = mesh->getNumVerts(), nf = mesh->getNumFaces();
		std::array<std::vector<Eigen::Triplet<double>>, 3> triplets;
		for (int i = 0; i < 3; i++) triplets[i].reserve(nf * 4);

		for (size_t i = 0; i < nf; i++) {
			auto normal = mesh->normal(mesh->faces[i]);
			normal.array() /= normal.squaredNorm();

			auto&& vidx = mesh->faces[i]->circuitVerts();
			for (size_t j = 0, k = vidx.size(); j < k; j++) {
				Eigen::Vector3d e = normal.cross(mesh->pos.row(vidx[(j+2)%k]) - mesh->pos.row(vidx[(j+1)%k]));
				for (int r = 0; r < 3; r++) triplets[r].emplace_back((int)i, (int)vidx[j], e(r));
			}
		}

		for (int i = 0; i < 3; i++) {
			grad[i].resize(nf, nv);
			grad[i].setFromTriplets(triplets[i].begin(), triplets[i].end());
		}

		return grad;
	}

	// divergence operator
	virtual const SparseMatrices3d& divergenceOp() const {
		if (div[0].size() != 0) return div;

		if (grad[0].rows() == 0) gradientOp();
		Eigen::VectorXd areas = mesh->faceAreas();

		for (int i = 0; i < 3; i++)
			div[i] = grad[i].transpose() * areas.asDiagonal();
		return div;
	}

	// curl operator
	virtual const SparseMatrices3d& curlOp() const {
		if (curl[0].rows() != 0) return curl;

		size_t nf = mesh->getNumFaces();
		std::array<std::vector<Eigen::Triplet<double>>, 3> triplets;
		for (int i = 0; i < 3; i++) triplets[i].reserve(nf);

		for (size_t i = 0; i < nf; i++) {
			auto normal = mesh->normal(mesh->faces[i]);
			normal.normalize();
			for (int j = 0; j < 3; j++) triplets[j].emplace_back((int)i, (int)i, normal[j]);
		}

		for (int i = 0; i < 3; i++) {
			curl[i].resize(nf, nf);
			curl[i].setFromTriplets(triplets[i].begin(), triplets[i].end());
		}

		return curl;
	}

	// barycentric areas (0-form hodge star)
	virtual const Eigen::VectorXd& barycentricAreas() const {
		if (hstar0.size() != 0) return hstar0;

		hstar0 = Eigen::VectorXd::Zero(mesh->getNumVerts());

		for (size_t i = 0; i < mesh->getNumFaces(); i++) {
			auto&& vidx = mesh->faces[i]->circuitVerts();
			double area = mesh->area(mesh->faces[i]);

			area /= (double)vidx.size();
			for (const auto& v : vidx) {
				hstar0[v] += area;
			}
		}

		return hstar0;
	}

	// voronoi areas for triangle mesh (0-form hodge star)
	virtual const Eigen::VectorXd& voronoiAreas() const {
		if (hstar0.size() != 0) return hstar0;

		hstar0 = Eigen::VectorXd::Zero(mesh->getNumVerts());

		for (size_t i = 0; i < mesh->getNumFaces(); i++) {
			double area = mesh->area(mesh->faces[i]);

			auto& e0 = mesh->faces[i]->edge;
			auto& e1 = e0->next;
			auto& e2 = e1->next;

			double a = mesh->length(e0), b = mesh->length(e1), c = mesh->length(e2);
			double a2 = a * a, b2 = b * b, c2 = c * c;
			bool isObtuse = (a2 + b2 + c2) < (2 * std::max({ a2, b2, c2 }));
			double cot0 = mesh->cot(e0), cot1 = mesh->cot(e1), cot2 = mesh->cot(e2);

			hstar0[e0->vert->id] += isObtuse ? (mesh->dot(e0->pair, e1) < 0.0 ? area * 0.5 : area * 0.25) :
				(a2 * cot0 + b2 * cot1) * 0.125;
			hstar0[e1->vert->id] += isObtuse ? (mesh->dot(e1->pair, e2) < 0.0 ? area * 0.5 : area * 0.25) :
				(b2 * cot1 + c2 * cot2) * 0.125;
			hstar0[e2->vert->id] += isObtuse ? (mesh->dot(e2->pair, e0) < 0.0 ? area * 0.5 : area * 0.25) :
				(c2 * cot2 + a2 * cot0) * 0.125;
		}

		return hstar0;
	}

	// vertex mass vector (0-form hodge star)
	virtual const Eigen::VectorXd& vertexAreas() const {
		return voronoiAreas();
	}

	// cotangent weights (1-form hodge star)
	virtual const Eigen::VectorXd& cotanWeights() const {
		if (hstar1.size() != 0) return hstar1;

		hstar1.resize(mesh->getNumEdges());

		for (size_t i = 0; i < mesh->getNumEdges(); i++) {
			auto& edge = mesh->edges[i];
			double a = mesh->cot(edge);
			double b = edge->pair->face == nullptr ? 0.0 : mesh->cot(edge->pair);
			hstar1[i] = 0.5 * (a + b);
		}

		return hstar1;
	}

	// inverse of face areas (2-form hodge star)
	virtual const Eigen::VectorXd& invFaceAreas() const {
		if (hstar2.size() != 0) return hstar2;
		hstar2.resize(mesh->getNumFaces());

		for (size_t i = 0; i < mesh->getNumFaces(); i++) {
			hstar2[i] = 1.0 / mesh->area(mesh->faces[i]);
		}

		return hstar2;
	}

	// vert to edge operator (0-form exterior derivative)
	virtual const Eigen::SparseMatrix<double>& verts2edgesOp() const {
		if (ederiv0.size() != 0) return ederiv0;
		size_t nv = mesh->getNumVerts(), ne = mesh->getNumEdges();

		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(ne * 2);

		for (int i = 0; i < (int)ne; i++) {
			auto& edge = mesh->edges[i];
			triplets.emplace_back(i, (int)edge->pair->vert->id, -1.0);
			triplets.emplace_back(i, (int)edge->vert->id, 1.0);
		}

		ederiv0.resize(ne, nv);
		ederiv0.setFromTriplets(triplets.begin(), triplets.end());

		return ederiv0;
	}

	// edge to face operator (1-form exterior derivative)
	virtual const Eigen::SparseMatrix<double>& edges2facesOp() const {
		if (ederiv1.size() != 0) return ederiv1;
		size_t ne = mesh->getNumEdges(), nf = mesh->getNumFaces();
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(nf * 3);

		for (int i = 0; i < (int)nf; i++) {
			auto e = mesh->faces[i]->edge, e0 = e;
			do {
				triplets.emplace_back(i, (int)e->uid(), e->isSecond() ? -1.0 : 1.0);
				e = e->next;
			} while (e != e0);
		}

		ederiv1.resize(nf, ne);
		ederiv1.setFromTriplets(triplets.begin(), triplets.end());
		return ederiv1;
	}

	// normalized surface gradient
	virtual Eigen::MatrixX3d integratedGradient(const Eigen::VectorXd& f) const {
		Eigen::MatrixX3d gf = grad * f;
		if (grad[0].rows() == 0) gradientOp();
		for (int i = 0; i < gf.rows(); i++)
			gf.row(i).stableNormalize();
		return gf;
	}

	// counter-clockwise ordered one-ring angles relative to reference edge(mesh->verts[i]->edge)
	virtual Eigen::SparseMatrix<double> oneringAngles() const {
		Eigen::SparseMatrix<double> phi(mesh->getNumVerts(), mesh->getNumVerts());
		std::vector<Eigen::Triplet<double>> ptriplets;
		ptriplets.reserve(6 * mesh->getNumVerts());

		for (size_t i = 0; i < mesh->verts.size(); i++) {
			auto oring = mesh->verts[i]->oneringNeighbs<heds::Idx, heds::CCW>();

			Eigen::VectorXd p(oring.size());

			Eigen::Vector3d v = mesh->pos.row(i);
			for (size_t j = 0, k = oring.size(); j < oring.size(); j++) {
				Eigen::Vector3d u = mesh->pos.row(oring[j]);
				Eigen::Vector3d w = mesh->pos.row(oring[(j+1)%k]);
				p[j] = kika::util::angle(u - v, w - v);
			}

			if (mesh->verts[i]->isBoundary() && ((p.sum() - p[0]) < std::numbers::pi)) {
				p[0] = 2.0 * std::numbers::pi - p[0];
			}

			p.array() *= 2.0 * std::numbers::pi / p.sum();

			double theta = 0.0;
			for (size_t j = 0; j < oring.size(); j++) {
				ptriplets.emplace_back((int)i, (int)oring[j], theta);
				theta += p[j];
			}
		}

		phi.setFromTriplets(ptriplets.begin(), ptriplets.end());
		return phi;
	}

	// counter-clockwise ordered one-ring edges (uniform length) relative to reference edge (mesh->verts[i]->edge)
	// pvf(j) = data(i, j) * vf(j); -> j to i parallel transport
	virtual Eigen::SparseMatrix<Eigen::dcomplex> oneringEdges() const {
		auto angles = oneringAngles();
		Eigen::SparseMatrix<Eigen::dcomplex> phi(mesh->getNumVerts(), mesh->getNumVerts());
		std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;
		triplets.reserve(mesh->edges.size() * 2);
		
		for (const auto& edge : mesh->edges) {
			auto [u, v] = edge->endpoint();
			Eigen::dcomplex ruv = std::polar(1.0, angles.coeff(u, v)), rvu = std::polar(1.0, angles.coeff(v, u));
			rvu = -rvu / ruv; ruv = std::conj(rvu);
			triplets.emplace_back(u, v, ruv);
			triplets.emplace_back(v, u, rvu);
		}
		phi.setFromTriplets(triplets.begin(), triplets.end());

		return phi;
	}

	// vertex-wise connection laplacian for triangle mesh
	virtual const Eigen::SparseMatrix<Eigen::dcomplex>& connectionLaplacianOp() const {
		if (clap.size() != 0) return clap;

		auto phi = oneringEdges();

		std::vector<Eigen::Triplet<Eigen::dcomplex>> triplets;
		triplets.reserve(9 * mesh->getNumFaces());

		for (const auto& face : mesh->faces) {
			auto& e0 = face->edge;
			auto& e1 = e0->next;
			auto& e2 = e1->next;

			double w0 = 0.5 * mesh->cot(e0);
			double w1 = 0.5 * mesh->cot(e1);
			double w2 = 0.5 * mesh->cot(e2);

			auto v0 = e0->vert->id;
			auto v1 = e1->vert->id;
			auto v2 = e2->vert->id;

			Eigen::dcomplex r01 = phi.coeff(v0, v1), r10 = phi.coeff(v1, v0);
			Eigen::dcomplex r12 = phi.coeff(v1, v2), r21 = phi.coeff(v2, v1);
			Eigen::dcomplex r02 = phi.coeff(v0, v2), r20 = phi.coeff(v2, v0);

			triplets.emplace_back(v0, v0, (w0 + w1));
			triplets.emplace_back(v1, v1, (w1 + w2));
			triplets.emplace_back(v2, v2, (w2 + w0));

			triplets.emplace_back(v0, v1, -w1 * r01);
			triplets.emplace_back(v1, v0, -w1 * r10);
			triplets.emplace_back(v1, v2, -w2 * r12);
			triplets.emplace_back(v2, v1, -w2 * r21);
			triplets.emplace_back(v2, v0, -w0 * r20);
			triplets.emplace_back(v0, v2, -w0 * r02);
		}

		clap.resize(mesh->getNumVerts(), mesh->getNumVerts());
		clap.setFromTriplets(triplets.begin(), triplets.end());
		return clap;
	}

};

}

namespace kika::diff {

/// @brief Discrete Differential Operators on Polygonal Meshes. [Goes D. F., et al. 2020].
class PixarPolyLaplacian : public DiscreteLaplacian {
	// disable
	const Eigen::VectorXd& voronoiAreas() const final { assert(!"Function Disabled"); return hstar0; }

	double lambda;

public:
	mutable std::vector<Eigen::MatrixXd> wlap; // face-wise laplace-bertrami operator (stiffness term)

	PixarPolyLaplacian(const double lambda = 1.0) : DiscreteLaplacian(), lambda(lambda) {}

	PixarPolyLaplacian(const heds::SurfaceMesh& mesh, const double lambda = 1.0) : DiscreteLaplacian(mesh), lambda(lambda) {}

	void build(const heds::SurfaceMesh& mesh) {
		this->mesh = &mesh;
		reset();
	}

	void reset() {
		wlap.resize(0);
		DiscreteLaplacian::reset();
	}

	const std::vector<Eigen::MatrixXd>& wlaplacianOp() const {
		if (wlap.size() != 0) return wlap;

		size_t nf = mesh->getNumFaces();
		wlap.resize(nf);
		
		auto& pos = mesh->pos;

		#pragma omp parallel for
		for (int i = 0; i < (int)nf; i++) {
			Eigen::Vector3d normal = 0.5 * mesh->normal(mesh->faces[i]);
			double area = normal.norm();
			Eigen::MatrixX3d X = pos(mesh->faces[i]->circuitVerts(), Eigen::all);
			normal.array() /= area;
			Eigen::Vector3d center = X.colwise().mean();
			auto D = eig::circuitMat(X.rows());
			Eigen::MatrixX3d B = 0.5 * (D.cwiseAbs() * X);
			Eigen::Matrix3Xd U = eig::skewSymMat(normal) * (B.transpose() - center.replicate(1, X.rows())) / area;
			Eigen::MatrixX3d E = D * X;
			Eigen::MatrixX3d V = E * (Eigen::Matrix3d::Identity() - normal * normal.transpose());
			Eigen::MatrixXd P = Eigen::MatrixXd::Identity(X.rows(), X.rows()) - V * U;
			Eigen::MatrixXd M = area * U.transpose() * U + lambda * P.transpose() * P;
			wlap[i] = D.transpose() * M * D;
		}

		return wlap;
	}

	const Eigen::SparseMatrix<double>& laplacianOp() const final {
		if (lap.size() != 0) return lap;
		if (wlap.size() == 0) wlaplacianOp();

		size_t nv = mesh->getNumVerts();
		
		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(mesh->getNumFaces() * 4 * 4);

		for (size_t i = 0; i < mesh->getNumFaces(); i++) {
			auto&& vidx = mesh->faces[i]->circuitVerts();
			for (size_t j = 0; j < vidx.size(); j++) {
				for (size_t k = 0; k < vidx.size(); k++) {
					triplets.emplace_back((int)vidx[k], (int)vidx[j], wlap[i](k, j));
				}
			}
		}

		lap.resize(nv, nv);
		lap.setFromTriplets(triplets.begin(), triplets.end());
		return lap;

		#if false
		//// prepare face to vertex matrix version
		// 
		//size_t nfv = (size_t)std::reduce(mesh->faces.begin(), mesh->faces.end(), 0ull, [](size_t v, auto& f) { return v + f->ngon(); });
		//
		//Eigen::SparseMatrix<double> v2f(nfv, nv), plap(nfv, nfv);
		//std::vector<Eigen::Triplet<double>> v2fTriples, plapTriples;
		//v2fTriples.reserve(nfv);
		//plapTriples.reserve(nfv);
		//
		//for (size_t i = 0, j = 0, k = 0; const auto& f : mesh->faces) {
		//	auto&& vidx = f->circuitVerts();
		//
		//	for (size_t t = 0; t < vidx.size(); t++) {
		//		//for (size_t r = 0; r < vidx.size(); r++) plapv2fTriples.emplace_back((int)(j+r), (int)vidx[t], wlap[k](r, t));
		//		v2fTriples.emplace_back((int)i++, (int)vidx[t], 1.0);
		//	}
		//
		//	for (size_t t = 0; t < vidx.size(); t++) {
		//		for (size_t r = 0; r < vidx.size(); r++) {
		//			plapTriples.emplace_back((int)(j+t), (int)(j+r), wlap[k](t, r));
		//		}
		//	}
		//
		//	j += vidx.size();
		//	k++;
		//}
		//
		//v2f.setFromTriplets(v2fTriples.begin(), v2fTriples.end());
		//plap.setFromTriplets(plapTriples.begin(), plapTriples.end());
		//
		//lap = v2f.transpose() * plap * v2f;
		//return lap;
		#endif
	}

	const Eigen::VectorXd& vertexAreas() const  final {
		return barycentricAreas();
	}

	const SparseMatrices3d& gradientOp() const final {
		if (grad[0].size() != 0) return grad;
	
		size_t nv = mesh->getNumVerts(), nf = mesh->getNumFaces();
		std::array<std::vector<Eigen::Triplet<double>>, 3> gtriplets, dtriplets;
		for (int i = 0; i < 3; i++) {
			gtriplets[i].reserve(nf * 4);
			dtriplets[i].reserve(nf * 4);
		}
		
		auto& pos = mesh->pos;
		for (int i = 0; i < (int)nf; i++) {
			Eigen::Vector3d normal = 0.5 * mesh->normal(mesh->faces[i]);
			double area = normal.norm();
			normal.array() /= area;
	
			auto&& verts = mesh->faces[i]->circuitVerts();
			Eigen::MatrixX3d X = pos(verts, Eigen::all);
			auto D = eig::circuitMat(X.rows());
			Eigen::MatrixX3d E = D * X;

			Eigen::Matrix3Xd G = - eig::skewSymMat(normal)* E.transpose() * 0.5 * D.cwiseAbs() / area;
			Eigen::MatrixX3d Div = G.transpose() * area;

			for (int j = 0; j < X.rows(); j++) {
				gtriplets[0].emplace_back(i, verts[j], G(0, j));
				gtriplets[1].emplace_back(i, verts[j], G(1, j));
				gtriplets[2].emplace_back(i, verts[j], G(2, j));
				dtriplets[0].emplace_back(verts[j], i, Div(j, 0));
				dtriplets[1].emplace_back(verts[j], i, Div(j, 1));
				dtriplets[2].emplace_back(verts[j], i, Div(j, 2));
			}
		}
		
		for (int i = 0; i < 3; i++) {
			grad[i].resize(nf, nv);
			grad[i].setFromTriplets(gtriplets[i].begin(), gtriplets[i].end());
			div[i].resize(nv, nf);
			div[i].setFromTriplets(dtriplets[i].begin(), dtriplets[i].end());
		}
		
		return grad;
	}

	const SparseMatrices3d& divergenceOp() const final {
		if (div[0].size() != 0) return div;
		if (grad[0].rows() == 0) gradientOp();

		return div;

		//// is this not good divergence operator? 
		//	size_t nv = mesh->getNumVerts(), nf = mesh->getNumFaces();
		//	std::array<std::vector<Eigen::Triplet<double>>, 3> triplets;
		//	for (int i = 0; i < 3; i++) triplets[i].reserve(nf * 4);
		//
		//	auto& pos = mesh->pos;
		//
		//	for (int i = 0; i < (int)nf; i++) {
		//		auto&& vidx = mesh->faces[i]->circuitVerts();
		//		// Div_f = D.T * M; <-----> Div_v = Div_f * E; (E = D * X);
		//		Eigen::MatrixX3d op = wlap[i] * pos(vidx, Eigen::all);
		//		for (int j = 0; j < (int)vidx.size(); j++) {
		//			for (int k = 0; k < 3; k++) {
		//				triplets[k].emplace_back((int)vidx[j], (int)i, op(j, k));
		//			}
		//		}
		//	}
		//
		//	for (int i = 0; i < 3; i++) {
		//		div[i].resize(nv, nf);
		//		div[i].setFromTriplets(triplets[i].begin(), triplets[i].end());
		//	}
		//
		//	return div;
	}

	// Eigen::MatrixX3d integratedGradient(const Eigen::VectorXd& f) const final {
	// if () {}
	// 	Eigen::VectorXd norm(mesh->getNumFaces());
	// 	for (size_t i = 0; i < mesh->getNumFaces(); i++) {
	// 		Eigen::VectorXd fi = f(mesh->faces[i]->circuitVerts());
	// 		norm[i] = std::sqrt((fi.transpose() * wlap[i] * fi).value() / mesh->area(mesh->faces[i]));
	// 	}
	// 	return (grad * f).array().colwise() / norm.array();
	// }

};

namespace impl {
class VirtualVertexSolver {
	Eigen::MatrixXd A;
	Eigen::VectorXd b;

public:
	VirtualVertexSolver() {}

	/// @brief to find a point that minimizing the area of the triangle fan.
	/// @param verts : counter-clockwise ordered verteices of a polygon.
	/// @return weight : (opt p = sigma {w_i verts_i}).
	Eigen::VectorXd solve4area(const Eigen::MatrixX3d& verts) {
		size_t n = (size_t)verts.rows();
		std::vector<Eigen::Vector3d> vdiff(n), vcross(n*n);
		for (size_t i = 0; i < n; i++) vdiff[i] = verts.row((i+1)%n) - verts.row(i);
		for (size_t i = 0; i < n; i++)
			for (size_t j = 0; j < n; j++) vcross[i * n + j] = verts.row(i).cross(vdiff[j]);

		A.resize(n+1, n);
		A.setZero();

		b.resize(n+1);
		b.setZero();

		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < n; j++) {
				for (size_t r = 0; r < n; r++) {
					//A(i, j) += (verts.row(j).cross(vdiff[r])).dot(verts.row(i).cross(vdiff[r]));
					A(i, j) += vcross[j * n + r].dot(vcross[i * n + r]);
				}
			}
		}

		A.array() *= 2.0;
		A.bottomRows(1).array() = 1.0;

		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < n; j++) {
				//b[i] +=	(verts.row(i).cross(vdiff[j])).dot(vdiff[j].cross(verts.row(j)));
				b[i] +=	vcross[i * n + j].dot(-vcross[j * n + j]);
			}
		}

		b.array() *= 2.0;
		b[n] = -1.0;

		// this will be switched jacobiSVD when matrix is small(<16)
		//return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-b);
		return A.completeOrthogonalDecomposition().solve(-b);
	}

	// centeroids
	Eigen::VectorXd solve4center(const Eigen::MatrixX3d& verts) {
		return Eigen::VectorXd::Constant(verts.rows(), 1.0 / verts.rows());
	}
};
}

// Polygon Laplacian Made Simple. [Bunge A., et al. 2020].
class VirtualVertexLaplacian : public DiscreteLaplacian {
	mutable heds::SurfaceMesh virtualMesh;
	mutable Eigen::SparseMatrix<double> P0, P2;

	//// UNDONE
	////void prolongationOp1() {
	////	size_t ne = mesh->getNumEdges(), vrne = virtualMesh.getNumEdges();
	////	std::vector<Eigen::Triplet<double>> triplets;
	////
	////	for (size_t i = 0; i < ne; i++) {
	////		auto [u, v] = mesh->edges[i]->endpoint();
	////
	////		for (Eigen::SparseMatrix<double>::InnerIterator r(P0, u); r; ++r) {
	////			double Pru = r.value();
	////			for (const auto& e : virtualMesh.verts[r.row()]->incidentEdges<heds::Ptr>()) {
	////				if (P0.coeff(e->pair->vert->id, v)) {
	////					triplets.emplace_back((int)e->uid(), (int)i, Pru * P0.coeff(e->pair->vert->id, v));
	////				}
	////			}
	////		}
	////	}
	////
	////	P1.resize(vrne, ne);
	////	P1.setFromTriplets(triplets.begin(), triplets.end());
	////}

	void computeProlongation() const {
		size_t nv = mesh->getNumVerts(), nf = mesh->getNumFaces(), nvf = nv + nf, ntri = 0;
		std::vector<Eigen::Triplet<double>> p0triples, p2triples;
		p0triples.reserve(nv + nf * 4);
		for (size_t i = 0; i < nv; i++) p0triples.emplace_back((int)i, (int)i, 1.0);
		p2triples.reserve(nf * 4);

		auto& pos = mesh->pos;
		Eigen::MatrixX3d virtualPos(nvf, 3);
		virtualPos.block(0, 0, nv, 3) = pos;
		std::vector<std::array<uint32_t, 3>> virtualTris;
		virtualTris.reserve(nf * 5);

		impl::VirtualVertexSolver vvs;

		for (size_t i = 0, t = 0; i < nf; i++) {
			auto&& vidx = mesh->faces[i]->circuitVerts();
			if (vidx.size() == 3) {
				ntri++;
				p2triples.emplace_back((int)virtualTris.size(), (int)i, 1.);
				virtualTris.emplace_back(std::array<uint32_t, 3>{ vidx[0], vidx[1], vidx[2] });
			} else {
				// construct triangle fun
				// insert vertex as j-th elements
				Eigen::MatrixX3d X = pos(vidx, Eigen::all);
				Eigen::VectorXd w = vvs.solve4area(X);
				double wf = 1. / vidx.size();

				size_t j = nv + t++;
				virtualPos.row(j) = X.transpose() * w;

				for (size_t r = 0; r < vidx.size(); r++) {
					p0triples.emplace_back((int)j, (int)vidx[r], w[r]);
					p2triples.emplace_back((int)virtualTris.size(), (int)i, wf);
					virtualTris.emplace_back(std::array<uint32_t, 3>{ vidx[r], vidx[(r + 1) % vidx.size()], (uint32_t)j });
				}
			}
		}

		virtualPos.conservativeResize(nvf - ntri, 3);

		P0.resize(virtualPos.rows(), nv);
		P0.setFromTriplets(p0triples.begin(), p0triples.end());

		P2.resize(virtualTris.size(), nf);
		P2.setFromTriplets(p2triples.begin(), p2triples.end());

		virtualMesh.build(virtualPos, virtualTris);
	}

public:

	VirtualVertexLaplacian() : DiscreteLaplacian() {}

	VirtualVertexLaplacian(const heds::SurfaceMesh& mesh) : DiscreteLaplacian(mesh) {}

	void build(const heds::SurfaceMesh& mesh) {
		this->mesh = &mesh;
		reset();
	}

	void reset() {
		P0.resize(0, 0);
		P2.resize(0, 0);
		virtualMesh.reset();
		DiscreteLaplacian::reset();
	}

	const Eigen::SparseMatrix<double>& laplacianOp() const  final {
		if (lap.size() != 0) return lap;
		if (P0.size() == 0) computeProlongation();
		lap = P0.transpose() * DiscreteLaplacian(virtualMesh).laplacianOp() * P0;
		return lap;
	}

	const SparseMatrices3d& gradientOp() const final {
		if (grad[0].size() != 0) return grad;
		if (P2.size() == 0) computeProlongation();
		grad = DiscreteLaplacian(virtualMesh).gradientOp();
		for (int i = 0; i < 3; i++)
			grad[i] = P2.transpose() * grad[i] * P0;
		return grad;
	}

	const SparseMatrices3d& divergenceOp() const final {
		if (div[0].size() != 0) return div;
		if (grad[0].rows() == 0) gradientOp();
		Eigen::VectorXd areas = P2.transpose() * virtualMesh.faceAreas();

		for (int i = 0; i < 3; i++)
			div[i] = grad[i].transpose() * areas.asDiagonal();
		return div;
	}

	const Eigen::VectorXd& barycentricAreas() const final {
		if (hstar0.size() != 0) return hstar0;
		if (P0.size() == 0) computeProlongation();
		hstar0 = P0.transpose() * DiscreteLaplacian(virtualMesh).barycentricAreas();
		return hstar0;
	}

	const Eigen::VectorXd& voronoiAreas() const final {
		if (hstar0.size() != 0) return hstar0;
		if (P0.size() == 0) computeProlongation();
		hstar0 = P0.transpose() * DiscreteLaplacian(virtualMesh).voronoiAreas();
		return hstar0;
	}

	const Eigen::VectorXd& vertexAreas() const final {
		return voronoiAreas();
	}

	const Eigen::VectorXd& invFaceAreas() const final {
		if (hstar2.size() != 0) return hstar2;
		if (P0.size() == 0) computeProlongation();
		hstar2 = P2.transpose() * DiscreteLaplacian(virtualMesh).invFaceAreas();
		return hstar2;
	}

};

// The Diamond Laplace for Polygonal and Polyhedral Meshes . [Bunge A., et al. 2021].
class DiamondLaplacian : public DiscreteLaplacian {
	// disable
	const Eigen::VectorXd& barycentricAreas() const final { assert(!"Function Disabled"); return hstar0; }
	const Eigen::VectorXd& voronoiAreas() const final { assert(!"Function Disabled"); return hstar0; }

	mutable Eigen::MatrixX3d dualPos;
	mutable Eigen::SparseMatrix<double> P;

	void computeDualProlongation() const {
		auto [nv, ne, nf] = mesh->vef();

		dualPos.resize(nf, 3);
		impl::VirtualVertexSolver vvs;

		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(nv + nf * 4);
		for (size_t i = 0; i < nv; i++) triplets.emplace_back((int)i, (int)i, 1.);
		for (size_t i = 0; i < nf; i++) {
			auto&& vidx = mesh->faces[i]->circuitVerts();
			Eigen::MatrixX3d X = mesh->pos(vidx, Eigen::all);
			auto w = vvs.solve4area(X);

			dualPos.row(i) = X.transpose() * w;
			for (size_t j = 0; j < vidx.size(); j++) 
				triplets.emplace_back((int)(i+nv), (int)vidx[j], w[j]);
		}

		P.resize(nv + nf, nv);
		P.setFromTriplets(triplets.begin(), triplets.end());
	}

	const Eigen::VectorXd& diamondAreas() const {
		if (hstar0.size() != 0) return hstar0;
		if (P.size() == 0) computeDualProlongation();

		auto [nv, ne, nf] = mesh->vef();
		auto& pos = mesh->pos;

		Eigen::VectorXd areas = Eigen::VectorXd::Zero(nv + nf);

		for (size_t i = 0; i < ne; i++) {
			auto& edge = mesh->edges[i];
			bool isBoundary = edge->isBoundary();

			auto [u, v] = edge->endpoint();
			auto [l, r] = edge->adjacentFaces();

			// diamond pos
			Eigen::Vector3d du = pos.row(u);
			Eigen::Vector3d dv = pos.row(v);
			Eigen::Vector3d dr = isBoundary ? 0.5 * (du + dv) : dualPos.row(r).transpose().eval();
			Eigen::Vector3d dl = dualPos.row(l);

			double area = 0.5 * (
				(dv - du).cross(dl - du).norm() + 
				(dr - du).cross(dv - du).norm()) / (4. - isBoundary);

			areas[u] += area;
			areas[v] += area;
			areas[nv + l] += area;
			if (!isBoundary) areas[nv + r] += area;
		}

		hstar0 = P.transpose() * areas;
		return hstar0;
	}

public:

	DiamondLaplacian() : DiscreteLaplacian() {}

	DiamondLaplacian(heds::SurfaceMesh& mesh) : DiscreteLaplacian(mesh) {}

	void build(const heds::SurfaceMesh& mesh) {
		this->mesh = &mesh;
		reset();
	}

	void reset() {
		P.resize(0, 0);
		dualPos.resize(0, 0);
		DiscreteLaplacian::reset();
	}

	const Eigen::SparseMatrix<double>& laplacianOp() const final {
		if (lap.size() != 0) return lap;
		if (P.size() == 0) computeDualProlongation();

		auto [nv, ne, nf] = mesh->vef();
		auto& pos = mesh->pos;

		Eigen::MatrixX2d diaAreas(ne, 2);
		std::array<std::vector<Eigen::Triplet<double>>, 2> triplets;
		for (int i = 0; i < 2; i++) triplets[i].reserve(ne * 5);

		for (size_t i = 0; i < ne; i++) {
			auto& edge = mesh->edges[i];
			bool isBoundary = edge->isBoundary();

			auto [u, v] = edge->endpoint();
			auto [l, r] = edge->adjacentFaces();

			// diamond pos
			Eigen::Vector3d du = pos.row(u);
			Eigen::Vector3d dv = pos.row(v);
			Eigen::Vector3d dr = isBoundary ? 0.5 * (du + dv) : dualPos.row(r).transpose().eval();
			Eigen::Vector3d dl = dualPos.row(l);

			// intrinsic diamond pos
			//   2
			// 4< >3
			//   1
			Eigen::Matrix<double, 2, 4> localG;

			Eigen::Vector3d e = (dv - du).normalized();
			Eigen::Vector3d n = (dl - dv).cross(dv - du).normalized();
			Eigen::Vector3d et = n.cross(e);

			Eigen::Vector2d dia1 = Eigen::Vector2d::Zero();
			Eigen::Vector2d dia2 = Eigen::Vector2d(0., (dv - du).norm());
			Eigen::Vector2d dia3 = isBoundary ? 0.5 * (dia1 + dia2) : Eigen::Vector2d((dr - du).dot(et), (dr - du).dot(e));
			Eigen::Vector2d dia4 = Eigen::Vector2d((dl - du).dot(et), (dl - du).dot(e));

			localG.row(0) << (dia3[1] - dia4[1]), (dia1[1] - dia2[1]), (dia4[1] - dia3[1]), (dia2[1] - dia1[1]);
			localG.row(1) << (dia4[0] - dia3[0]), (dia2[0] - dia1[0]), (dia3[0] - dia4[0]), (dia1[0] - dia2[0]);
			double denom = -localG.block<2, 2>(0, 0).determinant();
			if (abs(denom) < 1e-8) localG.setZero(); else localG.array() /= denom;

			double diaArea = 0.5 * (kika::eig::cross2(dia3, dia2) + kika::eig::cross2(dia2, dia4));

			for (int j = 0; j < 2; j++) {
				diaAreas(i, j) = diaArea;

				if (edge->isBoundary()) {
					triplets[j].emplace_back((int)i, (int)(v), 0.5 * localG(j, 3));
					triplets[j].emplace_back((int)i, (int)(u), 0.5 * localG(j, 3));
				} else {
					triplets[j].emplace_back((int)i, (int)(nv + r), localG(j, 3));
				}

				triplets[j].emplace_back((int)i, (int)(nv + l), localG(j, 1));
				triplets[j].emplace_back((int)i, (int)(u), localG(j, 0));
				triplets[j].emplace_back((int)i, (int)(v), localG(j, 2));
			}
		}

		Eigen::SparseMatrix<double> G(ne, nv + nf);
		grad.resize(ne, nv);
		div.resize(nv, ne);
		for (int i = 0; i < 2; i++) {
			G.setFromTriplets(triplets[i].begin(), triplets[i].end());
			grad[i] = G * P;
			div[i] = grad[i].transpose() * diaAreas.col(i).asDiagonal();
		}
		lap = div * grad;
		return lap;
	}

	const Eigen::VectorXd& vertexAreas() const final {
		return diamondAreas();
	}
		
	const SparseMatrices3d& gradientOp() const final {
		if (grad[0].size() != 0) return grad;
		if (grad[0].rows() == 0) laplacianOp();
		return grad;
	}

	const SparseMatrices3d& divergenceOp() const final {
		if (div[0].size() != 0) return div;
		if (div[0].rows() == 0) laplacianOp();
		return div;
	}

};


}
