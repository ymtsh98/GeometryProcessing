#pragma once

#include <Eigen/Core>
#include <numeric>

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

  AABB() : minCorner(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
    maxCorner(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()) {}

  AABB(const Eigen::MatrixX3d& pts) :
    minCorner(pts.colwise().minCoeff()),
    maxCorner(pts.colwise().maxCoeff()) {}

  void build(const Eigen::MatrixX3d& pts) {
    minCorner = pts.colwise().minCoeff();
    maxCorner = pts.colwise().maxCoeff();
  }

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

  bool intersects(const Eigen::Vector3d& center, const double r) const {
    // double d = 0.;
    // d += (center.array() >= minCorner.array()).select(0, (center.array() - minCorner.array()).square()).sum();
    // d += (center.array() <= maxCorner.array()).select(0, (center.array() - maxCorner.array()).square()).sum();
    // return (d <= r * r);
    double dmin = 0;
    for (int i = 0; i < 3; i++) {
      if (center[i] < minCorner[i]) dmin += (center[i] - minCorner[i]) * (center[i] - minCorner[i]);
      else if (center[i] > maxCorner[i]) dmin += (center[i] - maxCorner[i])* (center[i] - maxCorner[i]);
    }
    return (dmin <= r * r);
  }

  void expands(const AABB& other) {
    for (int i = 0; i < 3; i++) minCorner[i] = std::min(minCorner[i], other.minCorner[i]);
    for (int i = 0; i < 3; i++) maxCorner[i] = std::max(maxCorner[i], other.maxCorner[i]);
  }

  inline Eigen::Vector3d center() const {
    return 0.5 * (minCorner + maxCorner);
  }

  inline double center(int axis) const {
    return 0.5 * (minCorner[axis] + maxCorner[axis]);
  }

  // max or min
  template <bool Max>
  inline double distanceToCorner(const Eigen::Vector3d& p) const {

    return Max ? sqrt((minCorner - p).array().square().cwiseMax((maxCorner - p).array().square()).sum())
      : sqrt((minCorner - p).array().square().cwiseMin((maxCorner - p).array().square()).sum());
  }

  inline double surfaceArea() const {
    double x = (maxCorner[0] - minCorner[0]);
    double y = (maxCorner[1] - minCorner[1]);
    double z = (maxCorner[2] - minCorner[2]);
    return 2 * x * y + 2 * y * z + 2 * z * x;
  }

  inline double volume() const {
    return (maxCorner - minCorner).determinant();
  }

  std::pair<Eigen::MatrixX3d, std::vector<uint32_t>> drawableObject() const {
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

std::ostream& operator<<(std::ostream& out, const AABB& aabb) {
  out << aabb.minCorner.transpose() << "\n";
  out << aabb.maxCorner.transpose() << std::endl;
  return out;
}

}

namespace kika::spart {

// auto& [x, y, z] : grid.range<_>();
// for (int x = r1.x; x < r2.x; x++)
//   for (int y = r1.y; y < r2.y; y++)
//     for (int z = r1.z; z < r2.z; z++) 
template <typename T = int>
class GridRange {
  using Gtype = std::tuple<T, T, T>;

  struct GridIterator {
    using difference_type = std::ptrdiff_t;
    using value_type = Gtype;
    using reference = Gtype&;
    using pointer = Gtype*;
    using iterator_category = std::forward_iterator_tag;

    Gtype r;
    Gtype& r1, & r2;
    bool flag;

    GridIterator(Gtype& r1, Gtype& r2) : r(r1), r1(r1), r2(r2),
      flag(
      (std::get<0>(r2) > std::get<0>(r1)) &&
      (std::get<1>(r2) > std::get<1>(r1)) &&
      (std::get<2>(r2) > std::get<2>(r1))
      ) {}

    reference operator*() noexcept {
      return r;
    }

    GridIterator& operator++() {
      (std::get<2>(r2) > (std::get<2>(r) + 1)) ? ++std::get<2>(r) : (std::get<2>(r) = std::get<2>(r1),
        (std::get<1>(r2) > (std::get<1>(r) + 1)) ? ++std::get<1>(r) : (std::get<1>(r) = std::get<1>(r1),
        (std::get<0>(r2) > (std::get<0>(r) + 1)) ? ++std::get<0>(r) : flag = false));
      return *this;
    }

    bool operator!=(const GridIterator& itr) const {
      return flag;
    }

  };

  Gtype r1, r2;

public:

  GridRange(const int& r1, const int& r2) :
    r1((T)r1, (T)r1, (T)r1), r2((T)r2, (T)r2, (T)r2) {}

  GridRange(const int& r1, const Eigen::Vector3i r2) :
    r1((T)r1, (T)r1, (T)r1), r2((T)r2[0], (T)r2[1], (T)r2[2]) {}

  template <typename I>
  GridRange(const int& r1, const std::tuple<I, I, I>& r2) :
    r1((T)r1, (T)r1, (T)r1), r2((T)std::get<0>(r2), (T)std::get<1>(r2), (T)std::get<2>(r2)) {}

  GridRange(const Eigen::Vector3i& r1, const int& r2) :
    r1((T)r1[0], (T)r1[1], (T)r1[2]), r2((T)r2, (T)r2, (T)r2) {}

  template <typename I=int>
  GridRange(const std::tuple<I, I, I>& r1, const int& r2) :
    r1((T)std::get<0>(r1), (T)std::get<1>(r1), (T)std::get<2>(r1)), r2((T)r2, (T)r2, (T)r2) {}

  GridRange(const Eigen::Vector3i& r1, const Eigen::Vector3i& r2) :
    r1((T)r1[0], (T)r1[1], (T)r1[2]), r2((T)r2[0], (T)r2[1], (T)r2[2]) {}

  template <typename I=int>
  GridRange(const std::tuple<I, I, I>& r1, const std::tuple<I, I, I>& r2) :
    r1((T)std::get<0>(r1), (T)std::get<1>(r1), (T)std::get<2>(r1)),
    r2((T)std::get<0>(r2), (T)std::get<1>(r2), (T)std::get<2>(r2)) {}

  GridRange() : GridRange(0, 0) {}

  GridIterator begin() {
    return GridIterator(r1, r2);
  }

  GridIterator end() {
    return GridIterator(r1, r2);
  }

  T lx() { return std::get<0>(r1); }
  T rx() { return std::get<0>(r2); }
  T ly() { return std::get<1>(r1); }
  T ry() { return std::get<1>(r2); }
  T lz() { return std::get<2>(r1); }
  T rz() { return std::get<2>(r2); }
};
}

namespace kika::spart {

/// axis aligned sparse grid
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

    if (regular) {
      this->reso = ((aabb.maxCorner - aabb.minCorner).array() / stride.maxCoeff()).ceil().cast<int>() + 1;
      stride = Eigen::Vector3d::Constant(stride.maxCoeff());
    }

    for (int i = 0; i < 3; i++) this->reso[i] = std::max(this->reso[i], 2);
  }

  /// @param aabb : mesh aabb
  /// @param reso : grid resolution (num of grid points) >= 2
  /// @param regular : grid is regular or not
  AAGrid(const AABB& aabb, const int reso, const bool regular = false) : AAGrid(aabb, Eigen::Vector3i::Constant(reso), regular) {}

  // -------------------- grid operation --------------------

  // number of grid points
  template <axis Axis = axis::xyz>
  size_t size() const {
    return
      static_cast<size_t>(static_cast<int>(Axis) & 1 ? reso[0] : 1) *
      static_cast<size_t>(static_cast<int>(Axis) & 2 ? reso[1] : 1) *
      static_cast<size_t>(static_cast<int>(Axis) & 4 ? reso[2] : 1);
  }

  // grid position
  Eigen::Vector3d pos(const int x, const int y, const int z) const {
    return origin + Eigen::Vector3d(x * stride[0], y * stride[1], z * stride[2]);
  }

  // grid min corner
  Eigen::Vector3d minCorner() {
    return origin;
  }

  // grid max corner
  Eigen::Vector3d maxCorner() {
    return origin.array() + stride.array() * (reso.cast<double>().array() - 1.);
  }

  // -------------------- grid index --------------------

  size_t uniqueIndex(const int x, const int y, const int z) const {
    return static_cast<size_t>(x) * static_cast<size_t>(reso[2]) * static_cast<size_t>(reso[1]) +
      static_cast<size_t>(y) * static_cast<size_t>(reso[2]) +
      static_cast<size_t>(z);
  }

  size_t uniqueIndex(const std::tuple<int, int, int>& xyz) const {
    return uniqueIndex(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
  }

  // xy, yz, zx plane (a1 : x or y or z, a2 : y or z or x)
  template <axis Axis>
  size_t uniqueIndex(const int a1, const int a2) const {
    return static_cast<size_t>(reso[(static_cast<int>(Axis) / 2 + 1) % 3]) * static_cast<size_t>(a1) + static_cast<size_t>(a2);
  }

  size_t uniqueIndex(const Eigen::Vector3d& pos) const {
    Eigen::Vector3i xyz = ((pos - origin).array() / stride.array()).ceil().cast<int>() + 1;
    return uniqueIndex(xyz[0], xyz[1], xyz[2]);
  }

  std::tuple<int, int, int> gridIndex(const size_t uniqueId) const {
    return {
      static_cast<int>(uniqueId / (reso[1] * reso[2])),
      static_cast<int>((uniqueId / reso[2]) % reso[1]),
      static_cast<int>(uniqueId % reso[2]) };
  }

  // -------------------- cell operation --------------------

  // number of cells
  template <axis Axis = axis::xyz>
  size_t cellSize() const {
    return
      static_cast<size_t>(static_cast<int>(Axis) & 1 ? reso[0] - 1 : 1) *
      static_cast<size_t>(static_cast<int>(Axis) & 2 ? reso[1] - 1 : 1) *
      static_cast<size_t>(static_cast<int>(Axis) & 4 ? reso[2] - 1 : 1);
  }

  Eigen::Vector3d cellOffset(const int x, const int y, const int z) const {
    return Eigen::Vector3d(x * stride[0], y * stride[1], z * stride[2]);
  }

  Eigen::Vector3d cellOffset(const std::tuple<int, int, int>& xyz) const {
    return Eigen::Vector3d(std::get<0>(xyz) * stride[0], std::get<1>(xyz) * stride[1], std::get<2>(xyz) * stride[2]);
  }

  // -------------------- cell index --------------------

  // cell index corresponding to grid's index (max-corner) (x, y, z >= 1)
  size_t cellUniqueIndex(const int x, const int y, const int z) {
    return
      static_cast<size_t>(x - 1) * static_cast<size_t>(reso[2] - 1) * static_cast<size_t>(reso[1] - 1) +
      static_cast<size_t>(y - 1) * static_cast<size_t>(reso[2] - 1) +
      static_cast<size_t>(z - 1);
  }

  // cell index corresponding to grid's index (max-corner) (x, y, z >= 1)
  // xy, yz, zx plane (a1 : x or y or z, a2 : y or z or x)
  template <axis Axis>
  size_t cellUniqueIndex(const int a1, const int a2) const {
    return static_cast<size_t>(reso[(static_cast<int>(Axis) / 2 + 1) % 3] - 1) * static_cast<size_t>(a1 - 1) + static_cast<size_t>(a2 - 1);
  }

  // -------------------- range --------------------

  // auto& [x, y, z] : grid.range<_>(); (axis order is fixed)
  // for (int x = 0; x < reso.x; x++)
  //   for (int y = 0; y < reso.y; y++)
  //     for (int z = 0; z < reso.z; z++) 
  template <axis Axis = axis::xyz>
  GridRange<int> range() const {
    return GridRange<int>(0,
      { (static_cast<int>(Axis) & 1 ? reso[0] : 1),
       (static_cast<int>(Axis) & 2 ? reso[1] : 1),
       (static_cast<int>(Axis) & 4 ? reso[2] : 1) });
  }

  // aabb range (maximum containing grid)
  GridRange<int> range(const AABB& aabb, const double eps = 1e-16) const {
    return GridRange<int>(
      ((aabb.minCorner - origin).array() / stride.array()).floor().cast<int>(),
      (((aabb.maxCorner - origin).array() / stride.array() + eps).ceil().cast<int>()+1).min(reso.array()));
  }

  // qubic range (x-1, y-1, z-1 : x+1, y+1, z+1)
  GridRange<int> cellRange(const int x=1, const int y=1, const int z=1) const {
    return GridRange(Eigen::Vector3i(x-1, y-1, z-1), Eigen::Vector3i(x+1, y+1, z+1));
  }

};

}

namespace kika::spart {

class AABBTree {
public:

  std::vector<AABB> leaf;
  std::vector<AABB> tree;
  std::vector<uint64_t> children;
  std::vector<std::pair<uint32_t, uint32_t>> address;
  std::vector<uint32_t> indices;

  double timeAABB, timePoly;

  // compute SHA cost
  void splitSHA(const int axis, const double surfaceArea, std::vector<uint32_t>& ids, double& cost, size_t& splitSize, AABB& splitRight) {
    std::sort(ids.begin(), ids.end(), [&axis, this](uint32_t l, uint32_t r) {
      return leaf[l].center(axis) < leaf[r].center(axis);
    });

    AABB left, right;

    std::vector<double> surfaceAreaLeft(ids.size() - 1, 0.);
    for (size_t i = 0; i < (ids.size() - 1); i++) {
      left.expands(leaf[ids[i]]);
      surfaceAreaLeft[i] = left.surfaceArea();
    }

    for (size_t i = (ids.size() - 1); i != 0; i--) {
      right.expands(leaf[ids[i]]);
      double surfaceAreaRight = right.surfaceArea();

      double c = 2 * timePoly +
        (surfaceAreaLeft[i - 1] * i +
        surfaceAreaRight * (ids.size() - i)) *
        timeAABB / surfaceArea;

      if (c < cost) {
        cost = c;
        splitSize = i;
        splitRight = right;
      }
    }
  }

  // split space
  void split(const uint32_t space, std::vector<uint32_t> ids) {
    if (ids.size() == 0) return;

    double surfaceArea = tree[space].surfaceArea();

    double splitCost = timeAABB * ids.size();
    size_t splitSize = 0;
    AABB splitRight;
    std::vector<uint32_t> splitIds(ids.size());

    #pragma omp parallel for firstprivate(ids)
    for (int axis = 0; axis < 3; axis++) {

      double cost = splitCost;
      size_t size = 0;
      AABB right;

      splitSHA(axis, surfaceArea, ids, cost, size, right);

      #pragma omp critical
      {
        if (cost < splitCost) {
          splitCost = cost;
          splitSize = size;
          splitRight = right;
          splitIds = ids;
        }
      }
    }

    if (splitSize == 0) {
      address[space].first = (uint32_t)indices.size();
      address[space].second = (uint32_t)ids.size();
      indices.insert(indices.end(), ids.begin(), ids.end());

    } else {

      std::vector<uint32_t> leftIds(splitSize), rightIds(splitIds.size() - splitSize);
      for (size_t i = 0; i < leftIds.size(); i++) leftIds[i] = splitIds[i];
      for (size_t i = 0; i < rightIds.size(); i++) rightIds[i] = splitIds[splitIds.size() - i - 1];

      AABB splitLeft;
      for (size_t i = 0; i < leftIds.size(); i++) splitLeft.expands(leaf[splitIds[i]]);

      tree.emplace_back(splitLeft);
      children.push_back(0);
      address.emplace_back(0, 0);
      uint32_t l = static_cast<uint32_t>(tree.size() - 1);
      children[space] = static_cast<uint64_t>(l) << 32;
      split(l, leftIds);

      tree.emplace_back(splitRight);
      children.push_back(0);
      address.emplace_back(0, 0);
      uint32_t r = static_cast<uint32_t>(tree.size() - 1);
      children[space] |= static_cast<uint64_t>(r);
      split(r, rightIds);
    }

  }

public:
  AABBTree() : leaf({}), tree(1), children(1), address(1), indices({}), timeAABB(1.), timePoly(1.) {}

  template <typename T=std::vector<uint32_t>>
  AABBTree(const Eigen::MatrixX3d& pos, const std::vector<T>& tris) : AABBTree() {
    build(pos, tris);
  }

  template <typename T=std::vector<uint32_t>>
  void build(const Eigen::MatrixX3d& pos, const std::vector<T>& tris) {
    leaf.resize(tris.size());
    tree.reserve(2 * leaf.size());
    children.reserve(2 * leaf.size());
    address.reserve(2 * leaf.size());
    indices.reserve(leaf.size());

    for (size_t i = 0; i < tris.size(); i++) {
      leaf[i].build(pos(tris[i], Eigen::all));
      tree[0].expands(leaf[i]);
    }

    std::vector<uint32_t> ids(leaf.size());
    std::iota(ids.begin(), ids.end(), 0);

    split(0, ids);
  }

  // left child id
  uint32_t leftChild(const uint32_t p) const {
    return children[p] >> 32;
  }

  // right child id
  uint32_t rightChild(const uint32_t p) const {
    return children[p] & 0xffffffff;
  }

private:

  // closest point on surface
  // AutoShrink : shrink search range when search sphere hit any aabb
  // @param : eval(uint32_t candidate_index, Eigen::Vector3d query, const double range)
  template <bool AutoShrink, typename Func>
  void closestSurfacePointImpl(const Func& evalFunc, const Eigen::Vector3d& query, double& range, const uint32_t space) const {
    uint32_t lc = leftChild(space), rc = rightChild(space);

    if (lc == 0 && rc == 0) {
      if (!tree[space].intersects(query, range)) return;
      for (uint32_t i = 0, s = address[space].first; i < address[space].second; i++) evalFunc(indices[s + i], query, range);
      return;
    }

    if constexpr (AutoShrink) range = std::min(range, tree[space].distanceToCorner<true>(query));
    if (tree[space].intersects(query, range)) {
      closestSurfacePointImpl<AutoShrink>(evalFunc, query, range, lc);
      closestSurfacePointImpl<AutoShrink>(evalFunc, query, range, rc);
    }
  }

public:

  // closest point on surface
  // AutoShrink : shrink search range when search sphere hit any aabb
  // @param : eval(uint32_t candidate_index, Eigen::Vector3d query, const double range)
  template <bool AutoShrink = true, typename Func>
  void closestSurfacePoint(const Eigen::Vector3d& query, const Func& evalFunc, const double range = std::numeric_limits<double>::max()) const {
    double r = range;
    evalFunc(0, query, r);

    closestSurfacePointImpl<AutoShrink>(evalFunc, query, r, 0);
  }

};
}

// TODO : AABBTree
namespace kika::spart {
struct Line : public std::pair<double, double> {
  static constexpr int dim() {
    return 1;
  }

  Line() {}
  Line(const std::pair<double, double>& other) {
    build(other);
  }

  void build(const std::pair<double, double>& other) {
    this->first = other.first;
    this->second = other.second;
  }

  void expands(const Line& other) {
    this->first = std::min(this->first, other.first);
    this->second = std::max(this->second, other.second);
  }

  double center(int) const {
    return 0.5 * (this->first + this->second);
  }


  double surfaceArea() const {
    return (this->second - this->first);
  }

  bool contains(const double p) const {
    return first <= p && p <= second;
  }
};

struct Quad : public std::pair<std::pair<double, double>, std::pair<double, double>> {
  static constexpr int dim() {
    return 2;
  }

  Quad() {}
  Quad(const std::pair<std::pair<double, double>, std::pair<double, double>>& other) {
    build(other);
  }

  void build(const std::pair<std::pair<double, double>, std::pair<double, double>>& other) {
    this->first = other.first;
    this->second = other.second;
  }

  void expands(const Quad& other) {
    this->first = std::min(this->first, other.first);
    this->second = std::max(this->second, other.second);
  }

  double center(int axis) {
    if (axis == 0) return 0.5 * (first.first + first.second);
    else return 0.5 * (second.first + second.second);
  }


  double surfaceArea() {
    return (first.second - first.first) * (second.second - second.first);
  }

  template <typename T = std::pair<double, double>>
  bool contains(const T& p) const {
    return (first.first <= p.first && p.first <= first.second) &&
      (second.first <= p.second && p.second <= second.second);
  }
};

template <typename T>
class RangeTree {
  std::vector<T> leaf;
  std::vector<T> tree;
  std::vector<uint64_t> children;
  std::vector<std::pair<uint32_t, uint32_t>> address;
  std::vector<uint32_t> indices;

  double timeAABB, timePoly;

  // compute SHA cost
  void splitSHA(const int axis, const double surfaceArea, std::vector<uint32_t>& ids, double& cost, size_t& splitSize, T& splitRight) {
    std::sort(ids.begin(), ids.end(), [&axis, this](uint32_t l, uint32_t r) {
      return leaf[l].center(axis) < leaf[r].center(axis);
    });

    T left, right;

    std::vector<double> surfaceAreaLeft(ids.size() - 1, 0.);
    for (size_t i = 0; i < (ids.size() - 1); i++) {
      left.expands(leaf[ids[i]]);
      surfaceAreaLeft[i] = left.surfaceArea();
    }

    for (size_t i = (ids.size() - 1); i != 0; i--) {
      right.expands(leaf[ids[i]]);
      double surfaceAreaRight = right.surfaceArea();

      double c = 2 * timePoly +
        (surfaceAreaLeft[i - 1] * i +
        surfaceAreaRight * (ids.size() - i)) *
        timeAABB / surfaceArea;

      if (c < cost) {
        cost = c;
        splitSize = i;
        splitRight = right;
      }
    }
  }

  // split space
  void split(const uint32_t space, std::vector<uint32_t> ids) {
    if (ids.size() == 0) return;

    double surfaceArea = tree[space].surfaceArea();

    double splitCost = timeAABB * ids.size();
    size_t splitSize = 0;
    T splitRight;
    std::vector<uint32_t> splitIds(ids.size());

    #pragma omp parallel for firstprivate(ids)
    for (int axis = 0; axis < T::dim(); axis++) {

      double cost = splitCost;
      size_t size = 0;
      T right;

      splitSHA(axis, surfaceArea, ids, cost, size, right);

      #pragma omp critical
      {
        if (cost < splitCost) {
          splitCost = cost;
          splitSize = size;
          splitRight = right;
          splitIds = ids;
        }
      }
    }

    if (splitSize == 0) {
      address[space].first = (uint32_t)indices.size();
      address[space].second = (uint32_t)ids.size();
      indices.insert(indices.end(), ids.begin(), ids.end());

    } else {

      std::vector<uint32_t> leftIds(splitSize), rightIds(splitIds.size() - splitSize);
      for (size_t i = 0; i < leftIds.size(); i++) leftIds[i] = splitIds[i];
      for (size_t i = 0; i < rightIds.size(); i++) rightIds[i] = splitIds[splitIds.size() - i - 1];

      T splitLeft;
      for (size_t i = 0; i < leftIds.size(); i++) splitLeft.expands(leaf[splitIds[i]]);

      tree.emplace_back(splitLeft);
      children.push_back(0);
      address.emplace_back(0, 0);
      uint32_t l = static_cast<uint32_t>(tree.size() - 1);
      children[space] = static_cast<uint64_t>(l) << 32;
      split(l, leftIds);

      tree.emplace_back(splitRight);
      children.push_back(0);
      address.emplace_back(0, 0);
      uint32_t r = static_cast<uint32_t>(tree.size() - 1);
      children[space] |= static_cast<uint64_t>(r);
      split(r, rightIds);
    }

  }

public:
  RangeTree() : leaf({}), tree(1), children(1), address(1), indices({}), timeAABB(1.), timePoly(1.) {}

  template <typename Type>
  RangeTree(const std::vector<Type>& segments) : RangeTree() {
    build(segments);
  }

  template <typename Type>
  void build(const std::vector<Type>& segments) {
    leaf.resize(segments.size());
    tree.reserve(2 * leaf.size());
    children.reserve(2 * leaf.size());
    address.reserve(2 * leaf.size());
    indices.reserve(leaf.size());

    for (size_t i = 0; i < segments.size(); i++) {
      leaf[i].build(segments[i]);
      tree[0].expands(leaf[i]);
    }

    std::vector<uint32_t> ids(leaf.size());
    std::iota(ids.begin(), ids.end(), 0);

    split(0, ids);
  }

  // left child id
  uint32_t leftChild(const uint32_t p) const {
    return children[p] >> 32;
  }

  // right child id
  uint32_t rightChild(const uint32_t p) const {
    return children[p] & 0xffffffff;
  }

private:

  template <typename Query>
  void containIdsImpl(std::vector<uint32_t>& ids, const Query& query, const uint32_t space) const {
    uint32_t lc = leftChild(space), rc = rightChild(space);

    if (lc == 0 && rc == 0) {
      if (!tree[space].contains(query)) return;
      for (uint32_t i = 0, s = address[space].first; i < address[space].second; i++) {
        if (leaf[indices[s + i]].contains(query)) ids.emplace_back(indices[s + i]);
      }
      return;
    }

    if (tree[space].contains(query)) {
      containIdsImpl(ids, query, lc);
      containIdsImpl(ids, query, rc);
    }
  }

public:

  template <typename Query>
  std::vector<uint32_t> containIds(const Query& p) {
    std::vector<uint32_t> ids;
    containIdsImpl(ids, p, 0);
    return ids;
  }
};

}