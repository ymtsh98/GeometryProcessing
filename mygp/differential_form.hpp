#pragma once
namespace gp::diff {
// TODO : Form-0, 2 (extend polygonal mesh)
template <int Form>
Eigen::VectorXd hodgeStar(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
  size_t nv = mesh.getNumVerts(), ne = mesh.getNumEdges(), nf = mesh.getNumFaces();

  if constexpr (Form == 0) {
    return gp::lbo::voronoiAreas(mesh);
  } 

  Eigen::VectorXd star;

  if constexpr (Form == 1) {
    star.resize(ne);

    for (const auto& edge : mesh.edges) {
      double a = mesh.cot(edge);
      double b = edge->pair->face == nullptr ? 0.0 : mesh.cot(edge->pair);

      star[edge->id / 2] = 0.5 * (a + b);
    }

  } else if constexpr (Form == 2) {
    star.resize(nf);

    for (size_t i = 0; i < nf; i++) {
      star[i] = 1.0 / mesh.area(mesh.faces[i]);
    }
  }

  return star;
}

template <int Form>
Eigen::SparseMatrix<double> exteriorDerivative(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh) {
  size_t nv = mesh.getNumVerts(), ne = mesh.getNumEdges(), nf = mesh.getNumFaces();
  Eigen::SparseMatrix<double> deriv;
  std::vector<Eigen::Triplet<double>> triplets;

  if constexpr (Form == 0) {
    deriv.resize(ne, nv);
    triplets.reserve(ne * 2);

    for (const auto& edge : mesh.edges) {
      triplets.emplace_back((int)edge->id / 2, (int)edge->pair->vert->id, -1.0);
      triplets.emplace_back((int)edge->id / 2, (int)edge->vert->id, 1.0);
    }

  } else if constexpr (Form == 1) {
    deriv.resize(nf, ne);
    triplets.reserve(nf * 3);

    for (int i = 0; i < (int)nf; i++) {
      auto e = mesh.faces[i]->edge, e0 = e;
      do {
        triplets.emplace_back(i, (int)e->id / 2, e->id % 2 ? 1.0 : -1.0);
        e = e->next;
      } while (e != e0);
    }

  }

  deriv.setFromTriplets(triplets.begin(), triplets.end());
  return deriv;
}
}