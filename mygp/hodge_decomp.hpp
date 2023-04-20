#pragma once
#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/laplacian.hpp>
#include <mygp/differential_form.hpp>

namespace gp::field::impl {
Eigen::VectorXd exactComponent(
  const gp::heds::Mesh<Eigen::MatrixX3d>& mesh,
  const Eigen::VectorXd& df) {
  auto d0 = gp::diff::exteriorDerivative<0>(mesh);
  auto s1 = gp::diff::hodgeStar<1>(mesh);

  Eigen::SparseMatrix<double> A = d0.transpose() * s1.asDiagonal() * d0;
  Eigen::VectorXd b = d0.transpose() * s1.asDiagonal() * df;

  A.prune([](int i, int j, double val) { return (i == 0 || j == 0) ? 0. : val; });
  A.coeffRef(0, 0) = 1.0;
  b(0) = 0.;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
  Eigen::VectorXd exact = d0 * solver.solve(b);

  return exact;
}

Eigen::VectorXd coexactComponent(
  const gp::heds::Mesh<Eigen::MatrixX3d>& mesh,
  const Eigen::VectorXd& df) {

  auto d1 = gp::diff::exteriorDerivative<1>(mesh);
  auto s1 = gp::diff::hodgeStar<1>(mesh);

  Eigen::SparseMatrix<double> A = d1 * s1.cwiseInverse().asDiagonal() * d1.transpose();
  Eigen::VectorXd b = d1 * df;

  A.prune([](int i, int j, double val) { return (i == 0 || j == 0) ? 0. : val; });
  A.coeffRef(0, 0) = 1.0;
  b(0) = 0.;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
  Eigen::VectorXd coexact = s1.cwiseInverse().asDiagonal() * d1.transpose() * solver.solve(b);

  return coexact;
}
}

namespace gp::field {
// vertex wise vector field
std::tuple<Eigen::MatrixX3d, Eigen::MatrixX3d, Eigen::MatrixX3d> hodgeDecomp(
  const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& vectorField) {

  Eigen::VectorXd df(mesh.getNumEdges());
  for (size_t i = 0; const auto& edge : mesh.edges) {
    auto e = mesh.edge2vec(edge);
    df[i++] = 0.5 * (e.dot(vectorField.row(edge->vert->id)) + e.dot(vectorField.row(edge->pair->vert->id)));
    // f1[i++] = 0.5 * (e.dot(field.row(edge->face->id)) + e.dot(field.row(edge->pair->face->id))); //(face-wise field)
  }

  auto exact = impl::exactComponent(mesh, df);
  auto coexact = impl::coexactComponent(mesh, df);
  Eigen::VectorXd harmonic = df - exact - coexact;

  size_t nf = mesh.getNumFaces();
  Eigen::MatrixX3d exactField(nf, 3), coexactField(nf, 3), harmonicField(nf, 3);

  for (size_t i = 0; i < nf; i++) {
    auto n = mesh.normal(mesh.faces[i]);
    auto a = 0.5 * n.norm();

    Eigen::Matrix3d grad;
    int t = 0;
    auto e0 = mesh.faces[i]->edge, e = e0;
    do {
      Eigen::Vector3d g = n.cross(mesh.edge2vec(e)).normalized();
      grad.row(t++) = mesh.length(e) * g / (2. * a);
      e = e->next;
    } while (e != e0);

    t = 0;
    Eigen::Vector3d ex = Eigen::Vector3d::Zero();
    Eigen::Vector3d cx = Eigen::Vector3d::Zero();
    Eigen::Vector3d hr = Eigen::Vector3d::Zero();

    do {
      double sign = e->id % 2 ? 1.0 : -1.0;
      ex += sign * exact[e->id/2] *    (-grad.row((t+2)%3) + grad.row((t+1)%3));
      cx += sign * coexact[e->id/2] *  (-grad.row((t+2)%3) + grad.row((t+1)%3));
      hr += sign * harmonic[e->id/2] * (-grad.row((t+2)%3) + grad.row((t+1)%3));
      t++;
      e = e->next;
    } while (e != e0);

    exactField.row(i) = ex / 3.;
    coexactField.row(i) = cx / 3.;
    harmonicField.row(i) = hr / 3.;
  }

  // cvt to vertex-wise
  size_t nv = mesh.getNumVerts();
  Eigen::MatrixX3d exf = Eigen::MatrixX3d::Zero(nv, 3),
    cxf = Eigen::MatrixX3d::Zero(nv, 3),
    hrf = Eigen::MatrixX3d::Zero(nv, 3);
  for (size_t i = 0; i < mesh.getNumVerts(); i++) {
    auto n = mesh.normal(mesh.verts[i]);
    auto&& af = mesh.verts[i]->adjacentFindex();
    for (const auto& f : af) {
      exf.row(i).array() += gp::util::project(exactField.row(f), n).array();
      cxf.row(i).array() += gp::util::project(coexactField.row(f), n).array();
      hrf.row(i).array() += gp::util::project(harmonicField.row(f), n).array();
    }
    exf.row(i).array() /= (double)af.size();
    cxf.row(i).array() /= (double)af.size();
    hrf.row(i).array() /= (double)af.size();
  }

  return std::make_tuple(exf, cxf, hrf);
}
}