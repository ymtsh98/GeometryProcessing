#pragma once

#include <mygp/grid_range.hpp>

namespace gp::vox {

enum class axis : int {
  x = 1, 
  y = 2, 
  z = 4, 
  xy = 3,
  yz = 6, 
  zx = 5,
  xyz = 7,
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

  std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> cube() {
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

    ret.second = {
      0, 2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 7, 0, 1, 4, 4, 1, 5,
      0, 4, 2, 2, 4, 6, 2, 6, 3, 3, 6, 7, 3, 7, 1, 1, 7, 5 };
    
    return ret;
  }

};

/// axis aligned grid
class AAGrid {
public:
  Eigen::Vector3d origin;
  Eigen::Vector3i reso;
  Eigen::Vector3d stride;

  /// @param aabb : mesh aabb
  /// @param reso : grid resolution (num of grid points) >= 2
  AAGrid(const AABB& aabb, const Eigen::Vector3i& reso, const bool regular) :
    origin(aabb.minCorner), reso(reso),
    stride((aabb.maxCorner - aabb.minCorner).array() / (reso.array()-1).cast<double>()) {
    assert((reso.array() >= 1).all());

    if (regular) {
      this->reso = ((aabb.maxCorner - aabb.minCorner).array() / stride.maxCoeff()).ceil().cast<int>() + 1;
      stride = Eigen::Vector3d::Constant(stride.maxCoeff());
    }
  }

  /// @param aabb : mesh aabb
  /// @param reso : grid resolution (num of grid points) >= 2
  /// @param regular : grid is regular or not
  AAGrid(const AABB& aabb, const int reso, const bool regular = false) : AAGrid(aabb, Eigen::Vector3i::Constant(reso), regular) {}
  
  // number of grid points
  template <axis Axis = axis::xyz>
  int size() const {
    return (static_cast<int>(Axis) & 1 ? reso[0] : 1) * 
           (static_cast<int>(Axis) & 2 ? reso[1] : 1) * 
           (static_cast<int>(Axis) & 4 ? reso[2] : 1);
  }

  int uniqueIndex(const int x, const int y, const int z) const {
    return x * reso[2] * reso[1] + y * reso[2] + z;
  }

  // xy, yz, zx plane (a1 : x or y or z, a2 : y or z or x)
  template <axis Axis>
  int uniqueIndex(const int a1, const int a2) const {
    return reso[(static_cast<int>(Axis) / 2 + 1) % 3] * a1 + a2;
  }

  Eigen::Vector3d pos(const int x, const int y, const int z) const {
    return origin + Eigen::Vector3d(x * stride[0], y * stride[1], z * stride[2]);
  }

  Eigen::Vector3d minCorner() {
    return origin;
  }

  Eigen::Vector3d maxCorner() {
    return origin.array() + stride.array() * (reso.cast<double>().array() - 1.);
  }

  // -------------------- cell operation --------------------

  // number of cells
  template <axis Axis = axis::xyz>
  int cellSize() const {
    return (static_cast<int>(Axis) & 1 ? reso[0] - 1: 1) *
           (static_cast<int>(Axis) & 2 ? reso[1] - 1: 1) *
           (static_cast<int>(Axis) & 4 ? reso[2] - 1: 1);
  }

  Eigen::Vector3d cellOffset(const int x, const int y, const int z) const {
    return Eigen::Vector3d(x * stride[0], y * stride[1], z * stride[2]);
  }

  Eigen::Vector3d cellOffset(const std::tuple<int, int, int>& xyz) const {
    return Eigen::Vector3d(std::get<0>(xyz) * stride[0], std::get<1>(xyz) * stride[1], std::get<2>(xyz) * stride[2]);
  }

  // cell index corresponding to grid's index (max-corner) (x, y, z >= 1)
  int cellUniqueIndex(const int x, const int y, const int z) {
    return (x-1) * (reso[2] - 1) * (reso[1] - 1) + (y - 1) * (reso[2] - 1) + (z - 1);
  }

  // cell index corresponding to grid's index (max-corner) (x, y, z >= 1)
  // xy, yz, zx plane (a1 : x or y or z, a2 : y or z or x)
  template <axis Axis>
  int cellUniqueIndex(const int a1, const int a2) const {
    return (reso[(static_cast<int>(Axis) / 2 + 1) % 3] - 1) * (a1 - 1) + (a2 - 1);
  }

  // -------------------- range --------------------

  // auto [x, y, z] = grid.range<_>(); (axis order is fixed)
  // for (int x = 0; x < reso.x; x++)
  //   for (int y = 0; y < reso.y; y++)
  //     for (int z = 0; z < reso.z; z++) 
  template <axis Axis = axis::xyz>
  GridRange<int> range() const {
    return GridRange<int>(0, 
      {(static_cast<int>(Axis) & 1 ? reso[0] : 1),
       (static_cast<int>(Axis) & 2 ? reso[1] : 1),
       (static_cast<int>(Axis) & 4 ? reso[2] : 1) });
  }

  // aabb range
  GridRange<int> range(const AABB& aabb, const double eps = 1e-16) const {
    return GridRange<int>(
      ((aabb.minCorner - origin).array() / stride.array()).floor().cast<int>(),
      (((aabb.maxCorner - origin).array() / stride.array() + eps).ceil().cast<int>()+1).min(reso.array()));
  }

  // qubic range (x-1, y-1, z-1 : x+1, y+1, z+1)
  GridRange<int> cellRange(const int x=1, const int y=1, const int z=1) const {
    return GridRange(Eigen::Vector3i(x-1, y-1, z-1), Eigen::Vector3i(x+1, y+1, z+1));
  }

  /*
  AABB cell(const int x, const int y, const int z) const {
    Eigen::Matrix<double, 8, 3> cp;
    for (int i = 0; i < 8; i++) {
      cp.row(i) = gridPos(x + (i>>1&1^i&1)-1, y + (i>>1&1)-1, z + (i > 3) - 1);
    }
    return AABB(cp);
  }
  */

};
}