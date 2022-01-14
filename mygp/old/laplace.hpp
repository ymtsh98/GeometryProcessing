#if __has_include("Eigen/Eigen")
#define EIGEN_INCLUDE
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Sparse"
#endif

#include "geoproc.hpp"
#include "halfedge.hpp"

namespace gp {
	Eigen::SparseMatrix<double> cotanLaplacianSparseMat(const heds::Mesh& mesh,
																											const bool normalize = true) {
		using Triplet = Eigen::Triplet<double>;
		Eigen::SparseMatrix<double> mat(mesh.getNumVertices(), mesh.getNumVertices());
		std::vector<Triplet> triplets;

		Eigen::VectorXd sum_weights = Eigen::VectorXd::Zero(mesh.verts.size());
		for (auto& face : mesh.faces) {
			auto v0 = face->edge->vert;
			auto v1 = face->edge->next->vert;
			auto v2 = face->edge->next->next->vert;

			auto w0 = 0.5 * cot(mesh.edges.at({ v1, v2 }));
			auto w1 = 0.5 * cot(mesh.edges.at({ v2, v0 }));
			auto w2 = 0.5 * cot(mesh.edges.at({ v0, v1 }));

			triplets.push_back(Triplet(v0->id, v0->id, w2 + w1));
			triplets.push_back(Triplet(v1->id, v1->id, w2 + w0));
			triplets.push_back(Triplet(v2->id, v2->id, w0 + w1));

			triplets.push_back(Triplet(v0->id, v1->id, -w2));
			triplets.push_back(Triplet(v1->id, v0->id, -w2));
			triplets.push_back(Triplet(v1->id, v2->id, -w0));
			triplets.push_back(Triplet(v2->id, v1->id, -w0));
			triplets.push_back(Triplet(v2->id, v0->id, -w1));
			triplets.push_back(Triplet(v0->id, v2->id, -w1));

			sum_weights[v0->id] += (w2 + w1);
			sum_weights[v1->id] += (w2 + w0);
			sum_weights[v2->id] += (w1 + w0);
		}

		mat.setFromTriplets(triplets.begin(), triplets.end());
		if (normalize) {
			return  areaVectorPerVert(mesh).asDiagonal().inverse() * mat;
		}
		return mat;
	}
}