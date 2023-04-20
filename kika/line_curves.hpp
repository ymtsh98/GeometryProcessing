#pragma once

#include <kika/kikautil.hpp>

namespace kika::shape {
class LineCylinderizer {
public:
  double radius;
  bool isCycle;
  int reso;

  LineCylinderizer(const double radius = 1.0, const bool isCycle = true, const int reso = 16) : radius(radius), isCycle(isCycle), reso(reso) {}

  std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> generate(const Eigen::MatrixX3d& pos) {
    int reso = this->reso;
    size_t nv = (size_t)pos.rows();
    Eigen::MatrixX3d vert(reso * nv, 3);
    std::vector<uint32_t> idx(reso * 6 * nv);

    auto findMinVolume = [&idx, &vert, &reso, &nv](const auto i) {
      double minVolume = std::numeric_limits<double>::lowest();

      int offset1 = ((i == 0) ? (int)(nv-1) * reso : (i - 1) * reso);
      int offset2 = i * reso;
      std::vector<int> tmp(reso * 6);

      for (int start = 0; start < reso; start++) {
        for (int r = 0, c = 0; r < reso; r++, c+=6) {
          auto k = (r+start)%reso;
          tmp[c+0] = r + offset1;
          tmp[c+1] = k + offset2;
          tmp[c+2] = (r+1) % reso + offset1;
          tmp[c+3] = tmp[c+1];
          tmp[c+4] = (k+1)%reso + offset2;
          tmp[c+5] = tmp[c+2];
        }

        double vol = kika::util::computeVolume(vert, tmp);
        if (minVolume < vol) {
          minVolume = vol;
          for (size_t r = 0; r < reso * 6; r++) {
            idx[r + i * reso * 6] = static_cast<uint32_t>(tmp[r]);
          }
        }
      }
    };

    for (int i = 0, j = (int)(nv); i < j; i++) {
      Eigen::Vector3d p0 = pos.row(i == 0 ? (j - 1) : (i - 1));
      Eigen::Vector3d p1 = pos.row(i);
      Eigen::Vector3d p2 = pos.row((i + 1) % j);

      Eigen::Vector3d p = 0.5 * (p2 + p0) - p1, n, axis;
      if (p.squaredNorm() < 1e-16) {
        n = Eigen::Vector3d::Random();
        axis = p2 - p1;
      } else {
        // normal of Tri(p0, p1, p2)
        n = (p2 - p1).cross(-(p1 - p0)).normalized();
        axis = n.cross(p).normalized();
      }

      for (int r = 0; r < reso; r++) {
        double a = 2.0 * std::numbers::pi * ((double)r / reso);
        vert.row(r + i * reso) = kika::util::quatRot(radius * n, axis, a) + p1;
      }

      if (i == 0) continue;

      findMinVolume(i);
    }

    if (isCycle) findMinVolume(0);

    return { vert, idx };
  }
};
}