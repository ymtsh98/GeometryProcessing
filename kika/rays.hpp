
namespace kika::ray {
inline std::pair<double, double> rayIntersection(
  const Eigen::Vector2d& rs1, const Eigen::Vector2d& rd1, 
  const Eigen::Vector2d& rs2, const Eigen::Vector2d& rd2) {

  Eigen::Vector2d v1 = rs1 - rs2;
  Eigen::Vector2d v2 = rd2;
  Eigen::Vector2d v3 = Eigen::Vector2d(-rd1[1], rd1[0]);

  double cross21 = v2[0] * v1[1] - v2[1] * v1[0];

  return std::make_pair(cross21 / v2.dot(v3), v1.dot(v3) / v2.dot(v3));
}

}