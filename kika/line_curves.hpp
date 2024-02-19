#pragma once

#include <kika/kikautil.hpp>

namespace kika::shape {
class LineCylinderizer {
public:
  double radius;
  int reso;
  bool isCycle;

  LineCylinderizer(const double radius = 1.0, const bool isCycle = true, const int reso = 16) : radius(radius), reso(reso), isCycle(isCycle) {}

  std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> generate(const Eigen::MatrixX3d& pos) {
    int reso = this->reso;
    size_t nv = (size_t)pos.rows();
    Eigen::MatrixX3d vert(reso * nv, 3);
    std::vector<uint32_t> idx(reso * 6 * nv);

    auto findMaxVolume = [&idx, &vert, &reso, &nv](const auto i) {
      double maxVolume = std::numeric_limits<double>::lowest();

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

        double vol = fabs(kika::util::computeVolume(vert, tmp));
        if (maxVolume < vol) {
          maxVolume = vol;
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

      findMaxVolume(i);
    }

    if (isCycle) findMaxVolume(0);

    return { vert, idx };
  }

  std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> generate2(const Eigen::MatrixX3d& pos) {
    size_t nv = (size_t)pos.rows();
    Eigen::MatrixX3d vert(reso * nv, 3);
    std::vector<uint32_t> idx(reso * 6 * (nv - 1));

    auto findMaxVolume = [&idx, &vert, &nv, this](const auto i) {
      double maxVolume = std::numeric_limits<double>::lowest();

      int offset1 = (i - 1) * reso;
      int offset2 = i * reso;
      std::vector<int> tmp(reso * 6);
      
      for (int start = 0; start < reso; start++) {
        for (int r = 0, c = 0; r < reso; r++, c+=6) {
          auto k = (r+start)%reso;
          tmp[c+0] = r + offset1;
          tmp[c+1] = (r+1) % reso + offset1;
          tmp[c+2] = k + offset2;
      
          tmp[c+3] = tmp[c+1];
          tmp[c+4] = (k+1)%reso + offset2;
          tmp[c+5] = tmp[c+2];
        }
      
        double vol = fabs(kika::util::computeVolume(vert, tmp));
        if (maxVolume < vol) {
          maxVolume = vol;
          for (size_t r = 0; r < reso * 6; r++) {
            idx[r + offset1 * 6] = static_cast<uint32_t>(tmp[r]);
          }
        }

        for (auto p : tmp) {
          std::cout << p << ", ";
        }
        std::cout << std::endl;
      }

    };

    auto assignVert = [&vert, this](const Eigen::Vector3d& n, const Eigen::Vector3d& axis, const Eigen::Vector3d& p, const int offset) {
      for (int r = 0; r < reso; r++) {
        double angle = 2.0 * std::numbers::pi * ((double)r / reso);
        vert.row(r + offset) = kika::util::quatRot(radius * n, axis, angle) + p;
      }
    };

    {
      Eigen::Vector3d p0 = pos.row(0);
      Eigen::Vector3d p1 = pos.row(1);
      Eigen::Vector3d a = (p1 - p0).normalized();
      Eigen::Vector3d n = kika::util::orthogonal(a).normalized();
      Eigen::Vector3d axis = a;

      assignVert(n, axis, p0, 0);
    }

    for (int i = 1, j = (int)pos.rows(); i < j - 1; i++) {
      Eigen::Vector3d p0 = pos.row(i - 1);
      Eigen::Vector3d p1 = pos.row(i);
      Eigen::Vector3d p2 = pos.row(i + 1);

      Eigen::Vector3d p = 0.5 * (p2 + p0) - p1, n, axis;
      n = (p2 - p1).cross(-(p1 - p0)).normalized();
      axis = n.cross(p).normalized();

      assignVert(n, axis, p1, i * reso);

      findMaxVolume(i);
    }

    {
      Eigen::Vector3d p1 = pos.row(pos.rows() - 2);
      Eigen::Vector3d p2 = pos.row(pos.rows() - 1);
      Eigen::Vector3d a = (p2 - p1).normalized();
      Eigen::Vector3d n = kika::util::orthogonal(a).normalized();
      Eigen::Vector3d axis = a;

      assignVert(n, axis, p2, (pos.rows() - 1) * reso);
      findMaxVolume(pos.rows() - 1);
    }

    return { vert, idx };
  }
};
}