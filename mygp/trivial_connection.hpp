#pragma once
#include <mygp/alg_topology.hpp>
#include <mygp/differential_form.hpp>
#include <mygp/intrinsic_feature.hpp>
namespace gp::field::impl {
inline Eigen::VectorXd trivialHolonomy(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXd& areas,
  const Eigen::VectorXd& singularity, const Eigen::SparseMatrix<double>& deriv0, const Eigen::VectorXd& star1,
  const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver) {

  Eigen::VectorXd gc = -gp::feat::gaussianCurvature(mesh, areas).array() * areas.array();
  for (const auto& v : mesh.verts) {
    gc[v->id] += 2.0 * std::numbers::pi * singularity[v->id];
    gc[v->id] *= !v->isBoundary();
  }
  return star1.asDiagonal() * deriv0 * solver.solve(gc);
}

inline double localFrameDiff(const gp::heds::pte& e, const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& normals) {
  if (e->isBoundary()) return 0.0;

  Eigen::Vector3d se = mesh.edge2vec(e);
  Eigen::Vector3d le = mesh.edge2vec(e->face->edge).normalized();
  Eigen::Vector3d re = mesh.edge2vec(e->pair->face->edge).normalized();
  double dl = atan2(se.dot(normals.row(e->face->id).cross(le)), se.dot(le));
  double dr = atan2(se.dot(normals.row(e->pair->face->id).cross(re)), se.dot(re));
  return dl - dr;
}

inline Eigen::VectorXd nonTrivialHolonomy(
  const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& normals,
  const std::vector<std::vector<size_t>>& cycles,
  const Eigen::MatrixXd& harmonics,
  const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
  const Eigen::VectorXd& x) {

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
      b(i) += -localFrameDiff(e, mesh, normals);
    }

    b(i) = fmod(b(i), std::numbers::pi * 2.);
  }

  return H.lu().solve(b);
}

// dual cycles
template <typename T>
inline std::vector<std::vector<size_t>> nonContractibleCycles(const gp::heds::Mesh<T>& mesh) {
  std::vector<std::vector<size_t>> cycles;

  auto tct = gp::topo::treeCotreeDecomp(mesh, true, true);

  for (size_t i = 0; auto& cycle: tct) {
    cycles.emplace_back();
    cycles.reserve(cycle.size() * 2);

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

  return cycles;
}

// @param homologyBasis (dual edges).
inline Eigen::MatrixXd harmonicBasis(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh,
  const std::vector<std::vector<size_t>>& cycles,
  const Eigen::SparseMatrix<double>& d0, const Eigen::VectorXd& s1,
  const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver) {
  size_t g2 = cycles.size();

  Eigen::MatrixXd basis(mesh.getNumEdges(), g2);

  Eigen::SparseMatrix<double> delta = d0.transpose() * s1.asDiagonal();

  Eigen::VectorXd omega(basis.rows());
  for (size_t i = 0; i < g2; i++) {
    omega.fill(0.0);
    for (size_t j = 0; j < cycles[i].size(); j+=2) {
      auto e = mesh.halfedge((uint32_t)cycles[i][j], (uint32_t)cycles[i][j+1]);
      omega[e->id / 2] =  e->id % 2 ? 1.0 : -1.0;
    }

    basis.col(i) = s1.asDiagonal() * (omega - d0 * solver.solve(delta * omega));
  }
  return basis;
}

inline Eigen::MatrixX3d propagateAngle(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh,
  const Eigen::MatrixX3d normals,
  const Eigen::VectorXd& x,
  const Eigen::MatrixXd& harmonics,
  const Eigen::VectorXd& h, const size_t start = 0, const double initAngle = 0.0) {
  size_t nf = mesh.getNumFaces();
  size_t nb = (size_t)harmonics.cols();
  Eigen::MatrixX3d fields(nf, 3);

  for (size_t i = 0; i < nf; i++) {
    fields.row(i) = gp::util::quatRot(
      mesh.edge2vec(mesh.faces[i]->edge).normalized(), normals.row(i), initAngle);
  }

  std::vector<bool> check(nf, false);
  std::deque<size_t> queue;

  check[start] = true;
  queue.push_back(start);

  while (!queue.empty()) {
    auto i = queue.front();
    queue.pop_front();

    gp::heds::Halfedge* e0 = mesh.faces[i]->edge, * e = e0;
    do {
      if (e->pair->face == nullptr) { e = e->next; continue; }

      auto j = e->pair->face->id;

      if (check[j]) { e = e->next; continue; }

      Eigen::Vector3d sharedEdge = mesh.edge2vec(e).normalized();

      Eigen::Vector3d nij = normals.row(i).cross(normals.row(j));
      double dihedral = atan2(nij.norm(), normals.row(i).dot(normals.row(j)));
      dihedral *= sharedEdge.dot(nij) < 0.0 ? -1.0 : 1.0;

      double angle = (e->id % 2 ? 1.0 : -1.0) * x[e->id / 2];
      if (nb != 0) angle += (e->id % 2 ? 1.0 : -1.0) * harmonics(e->id / 2, Eigen::all).dot(h);

      Eigen::Vector3d u = gp::util::quatRot(fields.row(i), normals.row(i), -angle);
      fields.row(j) = gp::util::quatRot(u, sharedEdge, dihedral);

      check[j] = true;
      queue.push_back(j);

      e = e->next;
    } while (e != e0);
  }

  return fields;
}

}

namespace gp::field {
// convert vertex-wise vector field to face-wise vector field
inline Eigen::MatrixX3d cvt2facewise(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& vfield) {
  Eigen::MatrixX3d ffield = Eigen::MatrixX3d::Zero(mesh.getNumFaces(), 3);
  for (size_t i = 0; i < mesh.getNumFaces(); i++) {
    Eigen::Vector3d n = mesh.normal(mesh.faces[i]).normalized();
    auto&& av = mesh.faces[i]->adjacentVindex();
    for (const auto& v : av) {
      ffield.row(i).array() += util::project(vfield.row(v), n).array();
    }
    ffield.row(i).array() /= (double)av.size();
  }
  return ffield;
}

// convert face-wise vector field to vertex-wise vector field
inline Eigen::MatrixX3d cvt2vertexwise(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& ffield) {
  Eigen::MatrixX3d vfield = Eigen::MatrixX3d::Zero(mesh.getNumVerts(), 3);
  for (size_t i = 0; i < mesh.getNumVerts(); i++) {
    auto n = mesh.normal(mesh.verts[i]);
    auto&& af = mesh.verts[i]->adjacentFindex();
    for (const auto& f : af) {
      vfield.row(i).array() += util::project(ffield.row(f), n).array();
    }
    vfield.row(i).array() /= (double)af.size();
  }
  return vfield;
}


/// // v - e + f = - 2g + 2 - b 
/// (sometime : b = boundary - 1 ?)
/// Eigen::VectorXd singularity = Eigen::VectorXd::Zero(mesh.getNumVerts());
/// singularity[0] = 1, -1 ... etc;
/// 
/// compute trivial connections
/// @return vertex wise vector fields
inline Eigen::MatrixX3d trivialConnection(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh,
  const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, const Eigen::VectorXd& singularity, const double initAngle = 0.0, const bool fwise = false) {
  // ---------------------- prepare resources -----------------------
  Eigen::SparseMatrix<double> mass = gp::eig::spIdentity(areas.rows());
  mass.diagonal().array() *= 1e-8*areas.array();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(lbo+mass);

  size_t nf = mesh.getNumFaces();
  Eigen::MatrixX3d normals(nf, 3);
  for (size_t i = 0; i < nf; i++) {
    normals.row(i) = mesh.normal(mesh.faces[i]).normalized();
  }

  auto d0 = gp::diff::exteriorDerivative<0>(mesh);
  auto s1 = gp::diff::hodgeStar<1>(mesh);

  auto cycles = impl::nonContractibleCycles(mesh);
  Eigen::MatrixXd harmonics = impl::harmonicBasis(mesh, cycles, d0, s1, solver);

  //std::cout << mesh.eulerPoincare() << " " << harmonics.cols() << std::endl;

  // --------------------- trivial holonomy ----------------------------
  auto x = impl::trivialHolonomy(mesh, areas, singularity, d0, s1, solver);

  // --------------------- non trivial holonomy ------------------------
  auto h = impl::nonTrivialHolonomy(mesh, normals, cycles, harmonics, solver, x);

  // --------------------- propagate angles ---------------------------
  
  auto fields = impl::propagateAngle(mesh, normals, x, harmonics, h, 0, initAngle);
  if (fwise) return fields;

  Eigen::MatrixX3d vfields = cvt2vertexwise(mesh, fields);

  return vfields;
};

// convert extrinsic (vert-wise) vector field to intrinsic.
inline Eigen::VectorXcd cvt2intrinsic(const gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& vfield) {
  size_t n = mesh.getNumVerts();
  Eigen::VectorXcd intrin(n);

  for (size_t i = 0; i < n; i++) {
    auto n = mesh.normal(mesh.verts[i]);

    Eigen::Vector3d extrin = vfield.row(i);
    auto iedges = mesh.verts[i]->incidentEdges();
    std::reverse(iedges.begin()+1, iedges.end());

    double closestAngle = 10.0;
    size_t closestIndex = 0;
    for (size_t j = 0; j < iedges.size(); j++) {
      auto edge = mesh.edge2vec(iedges[j]->pair);
      auto angle = gp::util::angle(edge, extrin);
      if (closestAngle > angle) {
        closestAngle = angle;
        closestIndex = j;
      }
    }

    Eigen::Vector3d pedge = mesh.edge2vec(iedges[0]->pair);
    double oneringAngle = 0.0;
    for (size_t j = 1; j <= closestIndex; j++) {
      auto edge = mesh.edge2vec(iedges[j]->pair);
      oneringAngle += gp::util::angle(pedge, edge);
      pedge = edge;
    }

    oneringAngle += gp::util::angle(pedge, extrin, n);

    //intrin[i] = std::polar(mesh.length(mesh.verts[i]->edge), oneringAngle);
    intrin[i] = std::polar(vfield.row(i).norm(), oneringAngle);
  }

  return intrin;
}

}


/*
void traceStreamLine(
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& lines,
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& colors,
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& normals,
  std::vector<size_t>& index,
  size_t fi,
  const size_t maxIter,
  const double step, const double offset,
  const gp::heds::Mesh<Eigen::MatrixXd>& mesh,
  const std::unique_ptr<gp::spart::NanoFlannKNN>& centeroidTree,
  const Eigen::MatrixXd& fields,
  const int type = 0, const size_t seed = 0,
  const Eigen::Vector3d& startColor =  Eigen::Vector3d(0.5, 0.5, 1.0),
  const Eigen::Vector3d& endColor = Eigen::Vector3d(1.0, 0.1, 0.0)) {

  double totalStep = 0.0;
  Eigen::Vector3d v1, v2, v3, d1, d2, d3, n, v, u, bc, nbc;

  size_t iter = 0;
  for (; iter < maxIter;) {
    size_t s1 = mesh.faces[fi]->edge->vert->id;
    size_t s2 = mesh.faces[fi]->edge->next->vert->id;
    size_t s3 = mesh.faces[fi]->edge->next->next->vert->id;

    v1 = mesh.getPosition<Eigen::Vector3d>(s1);
    v2 = mesh.getPosition<Eigen::Vector3d>(s2);
    v3 = mesh.getPosition<Eigen::Vector3d>(s3);
    n = (v2 - v1).cross(v3 - v1).normalized();

    d1 = project(fields.row(s1), n);
    d2 = project(fields.row(s2), n);
    d3 = project(fields.row(s3), n);

    if (d1.dot(d2)* d2.dot(d3) < 0.5) break;

    if (type == 0) {}
    else if (type == 1) d1 = -d1, d2 = -d2, d3 = -d3;
    else if (type == 2) d1 = n.cross(d1), d2 = n.cross(d2), d3 = n.cross(d3);
    else if (type == 3) d1 = -n.cross(d1), d2 = -n.cross(d2), d3 = -n.cross(d3);

    if (iter == 0) {
      bc = Eigen::Vector3d(seed%3 == 0, seed%3 == 1, seed%3 == 2);
      v = v1 * bc[0] + v2 * bc[1] + v3 * bc[2];
    } else {
      v = project(v, n, v1);
      bc = barycoord(v, v1, v2, v3);
    }
    u = v;

    do {
      u += (d1 * bc[0] + d2 * bc[1] + d3 * bc[2]) * step;
      totalStep += (v - u).squaredNorm();
      lines.emplace_back(v + offset * n, u + offset * n);
      normals.emplace_back(n, n);
      v = u;
      bc = barycoord(u, v1, v2, v3);
      iter++;
    } while ((iter < maxIter) && (bc.array() >= 0.0).all());

    float dist{};
    if (iter < maxIter) centeroidTree->getNearestNeighbor(u.cast<float>(), fi, dist);
    if (dist > offset * 0.4) break;
  }

  if (iter < (size_t)(maxIter * 0.1)) {
    lines.resize(lines.size() - iter);
    normals.resize(lines.size());
    return;
  }

  double s = 0.0;
  Eigen::Vector3d c1 = endColor, c2;
  for (size_t i = 0, j = lines.size() - iter; i < iter; i++) {
    s += (lines[j + i].first - lines[j + i].second).squaredNorm();
    c2 = interpolate(endColor, startColor, s / totalStep);
    colors.emplace_back(c1, c2);
    c1 = c2;
  }

  index.emplace_back(lines.size());
}

std::tuple<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>,
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>,
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>,
std::vector<size_t>> drawableVectorFields(
  const size_t maxIter, const size_t maxStreams,
  const double step, const double offset,
  const gp::heds::Mesh<Eigen::MatrixXd>& mesh,
  const std::unique_ptr<gp::spart::NanoFlannKNN>& centeroidTree,
  const Eigen::MatrixXd& fields,
  const int nrosy = 1,
  const Eigen::Vector3d& startColor = Eigen::Vector3d(0.5, 0.5, 1.0),
  const Eigen::Vector3d& endColor = Eigen::Vector3d(1.0, 0.1, 0.0)) {

  std::random_device seed;
  std::mt19937 engine(seed());

  std::tuple<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>,
    std::vector<size_t>> streamsInfo;

  auto& lines = std::get<0>(streamsInfo);
  auto& colors = std::get<1>(streamsInfo);
  auto& normals = std::get<2>(streamsInfo);
  auto& index = std::get<3>(streamsInfo);
  lines.reserve(maxIter * maxStreams);
  colors.reserve(maxIter * maxStreams);
  normals.reserve(maxIter * maxStreams);
  index.reserve(maxStreams);

  for (size_t stream = 0; stream < maxStreams; stream++) {
    size_t fi = engine() % mesh.faces.size();

    int type = 0;

    if (nrosy == 2 && (stream % 2)) type = 1;

    if (nrosy == 4) {
      if (stream% 4 == 0) type = 1;
      else if (stream% 4 == 1) type = 2;
      else if (stream% 4 == 2) type = 3;
    }

    //auto sc = gp::color::randomColor<Eigen::Vector3d>(stream);
    //auto ec = gp::color::randomColor<Eigen::Vector3d>(stream);
    traceStreamLine(lines, colors, normals, index,
      fi, maxIter, step, offset, mesh, centeroidTree, fields, type, stream,
      startColor, endColor);
      //sc, ec);

  }

  return streamsInfo;
}
*/

/*
template <typename T>
std::vector<std::vector<size_t>> boundaryCycles(const gp::heds::Mesh<T>& mesh) {
  size_t nb = 0;
  std::vector<std::vector<size_t>> boundaries;
  std::vector<bool> inCycles(mesh.getNumEdges(), false);

  for (const auto& edge : mesh.edges) {
    if (!edge->isBoundary() || inCycles[edge->id / 2]) continue;

    boundaries.push_back({});

    auto e0 = edge, e = e0;
    do {
      boundaries[nb].emplace_back((size_t)e->vert->id);
      inCycles[e->id / 2] = true;

      e = e->next;
      while (!e->isBoundary()) {
        e = e->pair->next;
      };
    } while (e != e0);
    nb++;
  }

  return boundaries;
}

template <typename T>
std::vector<std::vector<size_t>> boundaryCells(const gp::heds::Mesh<T>& mesh) {
  size_t nb = 0;
  std::vector<std::vector<size_t>> boundaries;
  std::vector<bool> inCycles(mesh.getNumEdges(), false);

  for (const auto& edge : mesh.edges) {
    if (!edge->isBoundary() || inCycles[edge->id / 2]) continue;

    boundaries.push_back({});

    auto e0 = edge, e = e0;
    boundaries[nb].emplace_back((size_t)e->face->id);
    do {
      inCycles[e->id / 2] = true;

      e = e->next;
      while (!e->isBoundary()) {
        e = e->pair->next;
        boundaries[nb].emplace_back((size_t)e->face->id);
      };
    } while (e != e0);
    boundaries[nb].pop_back();
    nb++;
  }

  return boundaries;
}
*/
