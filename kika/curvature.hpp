#pragma once
#include <numbers>
#include <kika/kikautil.hpp>
#include <kika/diff_geometry.hpp>

namespace kika::curv {

inline Eigen::VectorXd meanCurvature(const heds::SurfaceMesh& mesh, const diff::DiscreteLaplacian& dsl, const Eigen::MatrixX3d& normals) {
  Eigen::MatrixX3d Lp = (0.5 * dsl.s0.cwiseInverse()).asDiagonal() * (dsl.lbo * mesh.pos);
  return (Lp.array() * normals.array()).rowwise().sum();
  // return (Lp.array() * normals.array()).rowwise().sum().cwiseSign().array() * (Lp.rowwise().norm()).array(); // (if normals is non-unit vectors)
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

	return curvature.array() / dsl.s0.array();
}

}