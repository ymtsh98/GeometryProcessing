#pragma once

#include <kika/eigutil.hpp>
#include <kika/surface_mesh.hpp>
#include <kika/curvature.hpp>
#include <kika/alg_topology.hpp>
#include <kika/diff_geometry.hpp>
#include <kika/vector_field.hpp>

namespace kika::vf {

class TrivialConnection {
  kika::heds::SurfaceMesh& mesh;
  kika::diff::DiscreteLaplacian& dsl;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  Eigen::MatrixX3d fnormals;
  std::vector<std::vector<uint32_t>> cycles;
  Eigen::MatrixXd harmonics;

  void harmonicBasis() {
    size_t g2 = cycles.size();

    harmonics.resize(mesh.getNumEdges(), g2);

    Eigen::SparseMatrix<double> delta = dsl.d0.transpose() * dsl.s1.asDiagonal();

    Eigen::VectorXd omega(harmonics.rows());
    for (size_t i = 0; i < g2; i++) {
      omega.fill(0.0);
      for (size_t j = 0; j < cycles[i].size(); j+=2) {
        auto e = mesh.halfedge((uint32_t)cycles[i][j], (uint32_t)cycles[i][j+1]);
        omega[e->id / 2] =  e->id % 2 ? 1.0 : -1.0;
      }

      harmonics.col(i) = dsl.s1.asDiagonal() * (omega - dsl.d0 * solver.solve(delta * omega));
    }

  }

  Eigen::VectorXd trivialHolonomy(const Eigen::VectorXd& singularity) {
    Eigen::VectorXd gc = -kika::curv::gaussianCurvature(mesh, dsl).array() * dsl.s0.array();
    for (const auto& v : mesh.verts) {
      gc[v->id] += 2.0 * std::numbers::pi * singularity[v->id];
      gc[v->id] *= !v->isBoundary();
    }
    return dsl.s1.asDiagonal() * dsl.d0 * solver.solve(gc);
  }

  double localFrameDiff(const kika::heds::pte& e) {
    if (e->isBoundary()) return 0.0;

    Eigen::Vector3d se = mesh.edgevec(e);
    Eigen::Vector3d le = mesh.edgevec(e->face->edge).normalized();
    Eigen::Vector3d re = mesh.edgevec(e->pair->face->edge).normalized();
    double dl = atan2(se.dot(fnormals.row(e->face->id).cross(le)), se.dot(le));
    double dr = atan2(se.dot(fnormals.row(e->pair->face->id).cross(re)), se.dot(re));
    return dl - dr;
  }

  Eigen::VectorXd nonTrivialHolonomy(const Eigen::VectorXd& x) {

    size_t nc = cycles.size();
    Eigen::VectorXd b = Eigen::VectorXd::Zero(nc);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nc, nc);

    for (size_t i = 0; i < nc; i++) {
      for (size_t k = 0; k < cycles[i].size(); k+=2) {
        auto e = mesh.halfedge((uint32_t)cycles[i][k], (uint32_t)cycles[i][k+1]);
        double sign = e->id % 2 ? 1.0 : -1.0;

        for (size_t j = 0; j < nc; j++) {
          H(i, j) += sign * harmonics(e->id / 2, j);
        }

        b(i) += -sign * x[e->id / 2];
        b(i) += -localFrameDiff(e);
      }

      b(i) = fmod(b(i), std::numbers::pi * 2.);
    }

    return H.lu().solve(b);
  }

  Eigen::MatrixX3d propagateAngle(
    const Eigen::VectorXd& x, const Eigen::VectorXd& h, const size_t start = 0, const double initAngle = 0.0) {
    size_t nf = mesh.getNumFaces();
    size_t nb = (size_t)harmonics.cols();
    Eigen::MatrixX3d fields(nf, 3);

    for (size_t i = 0; i < nf; i++) {
      fields.row(i) = kika::util::quatRot(
        mesh.edgevec(mesh.faces[i]->edge).normalized(), fnormals.row(i), initAngle);
    }

    std::vector<bool> arrived(nf, false);
    std::deque<size_t> queue;

    arrived[start] = true;
    queue.push_back(start);

    while (!queue.empty()) {
      auto i = queue.front();
      queue.pop_front();

      kika::heds::pte e0 = mesh.faces[i]->edge, e = e0;
      do {
        if (e->pair->face == nullptr) { e = e->next; continue; }

        auto j = e->pair->face->id;

        if (arrived[j]) { e = e->next; continue; }

        Eigen::Vector3d sharedEdge = mesh.edgevec(e).normalized();

        Eigen::Vector3d nij = fnormals.row(i).cross(fnormals.row(j));
        double dihedral = atan2(nij.norm(), fnormals.row(i).dot(fnormals.row(j)));
        dihedral *= sharedEdge.dot(nij) < 0.0 ? -1.0 : 1.0;

        double angle = (e->id % 2 ? 1.0 : -1.0) * x[e->id / 2];
        if (nb != 0) angle += (e->id % 2 ? 1.0 : -1.0) * harmonics(e->id / 2, Eigen::all).dot(h);

        Eigen::Vector3d u = kika::util::quatRot(fields.row(i), fnormals.row(i), -angle);
        fields.row(j) = kika::util::quatRot(u, sharedEdge, dihedral);

        arrived[j] = true;
        queue.push_back(j);

        e = e->next;
      } while (e != e0);
    }

    return fields;
  }

public:
  TrivialConnection(kika::heds::SurfaceMesh& mesh, kika::diff::DiscreteLaplacian& dsl) : mesh(mesh), dsl(dsl) {
    // ---------------------- prepare resources -----------------------
    dsl.laplacianOp();
    dsl.vertexAreas();
    dsl.cotanWeights();
    dsl.verts2edgesOp();

    size_t nf = mesh.getNumFaces();
    fnormals.resize(nf, 3);
    for (size_t i = 0; i < nf; i++) {
      fnormals.row(i) = mesh.normal(mesh.faces[i]).normalized();
    }

    solver.compute(dsl.lbo + kika::eig::spdiag(dsl.s0)  * 1e-8 );

    auto tct = kika::topo::treeCotreeDecomp<true>(mesh);
    cycles.resize(tct.size());
    for (size_t i = 0; auto& cycle: tct) {
      cycles[i].reserve(cycle.size() * 2);

      for (size_t j = 0, c = cycle.size(); j < c; j++) {
        auto e = mesh.faces[cycle[j]]->edge;

        while (true) {
          if (e->pair->face != nullptr && e->pair->face->id == cycle[(j+1)%c]) break;
          e = e->next;
        }
        cycles[i].emplace_back(e->vert->id);
        cycles[i].emplace_back(e->pair->vert->id);
      }

      i++;
    }

    harmonicBasis();
  }

  /// sum of singularity = v - e + f = - 2g + 2 - b 
  /// saddle (-1), tripod (-1/2),thorn(+1/2), focus(+1), apple(+3/2)
  /// Eigen::VectorXd singularity = Eigen::VectorXd::Zero(mesh.getNumVerts());
  /// singularity[i] = 1, -1 ... etc;
  /// 
  /// compute trivial connections
  /// @return face wise vector fields
  Eigen::MatrixX3d solve(const Eigen::VectorXd& singularity, const double initAngle = 0.0) {

    auto x = trivialHolonomy(singularity);

    auto h = nonTrivialHolonomy(x);

    return propagateAngle(x, h, 0, initAngle);
  }

  /// sum of singularityType = v - e + f = - 2g + 2 - b 
  /// saddle (-1), tripod (-1/2),thorn(+1/2), focus(+1), apple(+3/2)
  /// compute trivial connections
  /// @return face wise vector fields
  template <typename T = uint32_t>
  Eigen::MatrixX3d solve(const std::vector<T>& singularityIdx, const std::vector<double>& singularityType, const double initAngle = 0.0) {
    Eigen::VectorXd singularity = Eigen::VectorXd::Zero(mesh.getNumVerts());
    for (size_t i = 0; i < singularityIdx.size(); i++) {
      singularity[singularityIdx[i]] = singularityType[i];
    }

    return solve(singularity, initAngle);
  }

};

}