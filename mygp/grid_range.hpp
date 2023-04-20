#pragma once
#include <tuple>
namespace gp {

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
    Gtype &r1, &r2;
    bool flag;

    GridIterator(Gtype& r1, Gtype& r2) : r(r1), r1(r1), r2(r2),
      flag((std::get<0>(r2) > std::get<0>(r1)) &&
           (std::get<1>(r2) > std::get<1>(r1)) &&
           (std::get<2>(r2) > std::get<2>(r1))) {}

    reference operator*() noexcept {
      return r;
    }
  
    GridIterator& operator++() {
      (std::get<2>(r2) - 1 > std::get<2>(r)) ? ++std::get<2>(r) : (std::get<2>(r) = std::get<2>(r1), 
      (std::get<1>(r2) - 1 > std::get<1>(r)) ? ++std::get<1>(r) : (std::get<1>(r) = std::get<1>(r1),
      (std::get<0>(r2) - 1 > std::get<0>(r)) ? ++std::get<0>(r) : flag = false));
      return *this;
    }
  
    bool operator!=(const GridIterator& itr) const {
      return flag;
    }

  };

  Gtype r1, r2;
public:


  GridRange(const int r1, const int r2) : 
    r1((T)r1, (T)r1, (T)r1), r2((T)r2, (T)r2, (T)r2) {}

  GridRange(const int& r1, const Eigen::Vector3i& r2) :
    r1((T)r1, (T)r1, (T)r1), r2((T)r2[0], (T)r2[1], (T)r2[2]) {}

  template <typename I=int>
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