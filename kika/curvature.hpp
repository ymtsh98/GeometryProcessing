#pragma once
#include <numbers>
#include <kika/kikautil.hpp>
#include <kika/diff_geometry.hpp>

namespace kika::curv {

inline Eigen::VectorXd meanCurvature(const heds::SurfaceMesh& mesh, const diff::DiscreteLaplacian& dsl, const Eigen::MatrixX3d& vnormals) {
  Eigen::MatrixX3d Lp = (0.5 * dsl.vertexAreas().cwiseInverse()).asDiagonal() * (dsl.laplacianOp() * mesh.pos);
  return (Lp.array() * vnormals.array()).rowwise().sum();
  // return (Lp.array() * vnormals.array()).rowwise().sum().cwiseSign().array() * (Lp.rowwise().norm()).array(); // (if vnormals is non-unit vectors)
}

inline Eigen::VectorXd gaussianCurvature(const heds::SurfaceMesh& mesh, const diff::DiscreteLaplacian& dsl) {
  Eigen::VectorXd curvature(mesh.getNumVerts());
  for (size_t i = 0; i < mesh.getNumVerts(); i++) curvature[i] = (!mesh.verts[i]->isBoundary()+1) * std::numbers::pi;

  for (const auto& face : mesh.faces) {
    auto e0 = face->edge, e1 = e0->next;
    do {
      curvature[e0->vert->id] -= util::angle(-mesh.edgevec(e0), mesh.edgevec(e1));
      e0 = e1;
      e1 = e1->next;
    } while (e0 != face->edge);
  }

	return curvature.array() / dsl.vertexAreas().array();
}

// @return facewise total curvature
inline Eigen::VectorXd totalCurvature(const heds::SurfaceMesh& mesh, const diff::PixarPolyLaplacian& dsl, const Eigen::MatrixX3d& vnormals) {
  int n = (int)mesh.getNumFaces();
  Eigen::VectorXd curvature(n);

  dsl.laplacianOp();
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    Eigen::Matrix3Xd N = vnormals(mesh.faces[i]->circuitVerts(), Eigen::all).transpose();
    curvature[i] = (N * dsl.weakLbo[i] * N.transpose()).trace();
  }

  return curvature;
}

}