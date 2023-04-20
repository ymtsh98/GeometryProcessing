#pragma once

#include <Eigen/Core>

namespace kika::spart {

enum class axis : int {
  x   = 0b001,
  y   = 0b010,
  z   = 0b100,
  xy  = 0b011,
  yz  = 0b110,
  zx  = 0b101,
  xyz = 0b111,
};

class AABB {
public:
  Eigen::Vector3d minCorner, maxCorner;

  AABB() : minCorner({}), maxCorner({}) {}

  AABB(const Eigen::MatrixX3d& pts) :
    minCorner(pts.colwise().minCoeff()),
    maxCorner(pts.colwise().maxCoeff()) {}

  bool contains(const Eigen::Vector3d& point) const {
    return ((minCorner.array() <= point.array()) &&
      (point.array() <= maxCorner.array())).all();
  }

  bool contains(const AABB& other) const {
    return ((minCorner.array() <= other.minCorner.array()) &&
      (maxCorner.array() >= other.maxCorner.array())).all();
  }

  bool intersects(const AABB& other) const {
    return ((minCorner.array() <= other.maxCorner.array()) &&
      (other.minCorner.array() <= maxCorner.array())).all();
  }

  Eigen::Vector3d center() {
    return 0.5 * (minCorner + maxCorner);
  }

  std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> drawableObject() {
    std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> ret(Eigen::MatrixX3d(8, 3), std::vector<uint32_t>(36));
    ret.first <<
      maxCorner[0], minCorner[1], minCorner[2],
      maxCorner[0], minCorner[1], maxCorner[2],
      maxCorner[0], maxCorner[1], minCorner[2],
      maxCorner[0], maxCorner[1], maxCorner[2],
      minCorner[0], minCorner[1], minCorner[2],
      minCorner[0], minCorner[1], maxCorner[2],
      minCorner[0], maxCorner[1], minCorner[2],
      minCorner[0], maxCorner[1], maxCorner[2];

    ret.second ={
      0, 2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 7, 0, 1, 4, 4, 1, 5,
      0, 4, 2, 2, 4, 6, 2, 6, 3, 3, 6, 7, 3, 7, 1, 1, 7, 5 };

    return ret;
  }

};

}