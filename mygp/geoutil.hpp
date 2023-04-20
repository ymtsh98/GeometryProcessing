#pragma once
#include <iostream>
#include <algorithm>
#include <numeric>
#include <numbers>
#include <vector>
#include <chrono>
#include <random>
#include <complex>
#include <functional>

#if __has_include("Eigen/Eigen")
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif

/// 
/// concept and SFINAE
/// 
namespace gp::util {

template <typename T>
concept IsEigen = requires(T t) {
  t.rows;
  t.cols;
};

template <typename T>
concept IsGlm = requires(T t) {
  t.x;
  t.r;
  t.length;
};

template <typename T>
concept IsStdvec = requires(T t) {
  t.push_back;
  t.shrink_to_fit;
};

template <typename T>
concept IsStdarr = requires(T t) {
	t.max_size;
};


template <typename T>
concept IsStdvec2d = requires(T t) {
  IsStdvec<T>;
  t[0][0];
};

template <typename T>
concept IsScalar = std::is_scalar_v<T>;

/*
template <typename T, class = void>
struct IsEigen : std::false_type {};

template <typename T>
struct IsEigen<T, std::void_t<decltype(std::declval<T>().rows())>> : std::true_type {};

template <typename T, class = void>
struct IsStdvec : std::false_type {};

template <typename T>
struct IsStdvec<T, std::void_t<decltype(std::declval<T>().push_back())>> : std::true_type {};

template <typename T, class = void>
struct IsStdvec2d : std::false_type {};

template <typename T>
struct IsStdvec2d<T, std::void_t<decltype(std::declval<T>()[0][0])>> : std::true_type {};

template <typename T, class = void>
struct IsStdarr : std::false_type {};

template <typename T>
struct IsStdarr<T, std::void_t<decltype(std::declval<T>().max_size())>> : std::true_type {};
*/

}

/// 
/// basic utility
/// 
namespace gp::util {
template <typename T>
size_t getRows(const T& a) {
  if constexpr (IsEigen<T>) {
    return static_cast<size_t>(a.rows());
  }
  return a.size();
}

template <typename T>
size_t getCols(const T& a) {
  if constexpr (IsEigen<T>) {
    return static_cast<size_t>(a.cols());
  }
  return a[0].size();
}

template <typename I=size_t>
std::vector<I> vecrange(const size_t n, const I init = 0) {
	std::vector<I> vec(n);
	std::iota(vec.begin(), vec.end(), init);
	return vec;
}

template <typename I=size_t>
std::vector<I> randomPermutation(const size_t n) {
  std::random_device seed;
  std::mt19937 engine(seed());

  std::vector<I> randomIndices(n);
  std::iota(randomIndices.begin(), randomIndices.end(), 0);
  std::shuffle(randomIndices.begin(), randomIndices.end(), engine);

  return randomIndices;
}

inline double mapping(const double value, const double start1, const double stop1,
  const double start2 = 0.0, const double stop2 = 1.0) {
  return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}

template <typename T1, typename T2>
void wsort(std::vector<T1>& v, std::vector<T2>& weights) {
	std::vector<int> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&weights](int a, int b) { return weights[a] < weights[b]; });

	std::vector<T1> rv = v;
	std::vector<T2> rw = weights;
	std::transform(idx.begin(), idx.end(), rv.begin(), [&v](int i){ return v[i]; });
	std::transform(idx.begin(), idx.end(), rw.begin(), [&weights](int i){ return weights[i]; });
	v = rv;
	weights = rw;
}

template <typename T1, typename T2>
std::vector<T1> vecast(const std::vector<T2>& vec) {
	return std::vector<T1>(vec.begin(), vec.end());
}

/// operation push_back as increment counter
/// usage :
/// 
/// std::set<T> s1 = {...}, s2 = {...};
/// push_back_counter<uint32_t> pbc;
/// std::set_union(s1.begin(), s1.end(), s2.begin(), 
///		s2.end(), std::back_inserter(pbc));
/// std::cout << pbc.count << std::endl;
/// 
template <typename T>
struct push_back_counter {
public:
	using value_type = T;

	void push_back(T const&) { ++count; }

	std::size_t count = 0;
};

}

///
/// mathematics
/// 
namespace gp::util {
template <typename T>
inline T pow2(const T& a) {
  return a * a;
}

template <typename T>
inline T zeroBoundSqrt(const T& a) {
	return (a >= 0) ? sqrt(a) : 0;
}

template <typename T>
T l2norm(const std::vector<T>& vec) {
	T sum = 0.0;
	for (size_t i = 0; i < vec.size(); i++) sum += (vec[i] * vec[i]);
	return sqrt(sum);
}

template <typename T>
inline T sign(const T& a) {
	return (a>0)-(a<0);
}
}

///
/// geometry
/// 
namespace gp::util {
template<typename T1, typename T2, typename T3>
inline double cot(const T1& a, const T2& b, const T3& c) {
	if constexpr (IsEigen<T1>) {
    #ifdef EIGEN_MAJOR_VERSION
		Eigen::Vector3<typename T1::value_type> e1 = b - a;
		Eigen::Vector3<typename T1::value_type> e2 = c - a;
		double dot = e1.dot(e2);
		double crossNorm = e1.cross(e2).norm();
		return dot / crossNorm;
    #endif
	} else {
		double dot = (b[0] - a[0]) * (c[0] - a[0]) +
			(b[1] - a[1]) * (c[1] - a[1]) +
			(b[2] - a[2]) * (c[2] - a[2]);
    double cross0 = (b[1] - a[1])* (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    double cross1 = (b[2] - a[2])* (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    double cross2 = (b[0] - a[0])* (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
		return dot / sqrt(cross0 * cross0 + cross1 * cross1 + cross2 * cross2);
	}
}

inline bool isObtuseTriangle(double a, double b, double c) {
  if (a > b) std::swap(a, b);
  if (b > c) std::swap(b, c);
  if (a > b) std::swap(a, b);

  return (c * c) > (a * a + b * b);
}

template <typename T, typename F = T::value_type>
T normal(const T& a, const T& b, const T& c) {
	return T{ (b[1] - a[1])* (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
					(b[2] - a[2])* (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
					(b[0] - a[0])* (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) };
}

template <typename T1, typename T2=size_t, typename T3= typename T1::value_type>
std::vector<T1> vertexNormals(const std::vector<T1>& pos, const std::vector<T2>& faces) {
	std::vector<T1> normals;
	if constexpr (IsEigen<T1>) normals.resize(pos.size(), T1::Zero(3));
	else normals.resize(pos.size(), T1{ (T3)0., (T3)0., (T3)0. });

	if constexpr (IsStdvec<T2> || IsStdarr<T2>) {
		for (size_t i = 0; i < faces.size(); i++) {
			auto n = normal(pos[faces[i][0]], pos[faces[i][1]], pos[faces[i][2]]);
			for (size_t r = 0; r < 3; r++) {
				for (size_t j = 0; j < 3; j++) {
					normals[faces[i][r]][j] += n[j];
					normals[faces[i][r]][j] += n[j];
					normals[faces[i][r]][j] += n[j];
				}
			}
		}
	} else {
		for (size_t i = 0; i < faces.size(); i+=3) {
			auto n = normal(pos[faces[i]], pos[faces[i+1]], pos[faces[i+2]]);
			for (size_t j = 0; j < 3; j++) {
				normals[faces[i]][j] += n[j];
				normals[faces[i+1]][j] += n[j];
				normals[faces[i+2]][j] += n[j];
			}
		}
	}
	for (size_t i = 0; i < pos.size(); i++) {
		double norm = l2norm(normals[i]);
		for (size_t j = 0; j < 3; j++) normals[i][j] /= norm;
	}
	return normals;
}

#ifdef EIGEN_MAJOR_VERSION
// signed volume (triangle mesh only)
inline double computeVolume(const Eigen::MatrixX3d& verts, const std::vector<uint32_t>& idx) {
	double volume = 0.;
	for (size_t i = 0; i < idx.size(); i+=3) {
		volume += verts({ idx[i], idx[i+1], idx[i+2] }, Eigen::all).determinant();
	}
	return (1. / 6.) * volume;
}

// unsigned angle
inline double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  //return atan2(a.cross(b).norm(), a.dot(b));
  return acos(std::min(std::max(a.dot(b) / (a.norm() * b.norm()), -1.0), 1.0));
}

// signed angle (right-handed)
inline double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& n) {
	return atan2(a.cross(b).dot(n), a.dot(b));
	//return sign(a.cross(b).dot(n)) * acos(std::min(std::max(a.dot(b) / (a.norm() * b.norm()), -1.0), 1.0));
}

// rotate v 
inline Eigen::Vector3d quatRot(const Eigen::Vector3d& v, const Eigen::Vector3d& axis, double angle) {
	angle *= 0.5;
	Eigen::Quaternion q(cos(angle), sin(angle) * axis(0), sin(angle) * axis(1), sin(angle) * axis(2));
	return q * v;
}

// barycentric coord s on triangle(a, b, c)
inline Eigen::Vector3d barycoord(const Eigen::Vector3d& s, const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
	Eigen::Vector3d d1 = b - a, d2 = c - a, d3 = s - a;
	double d1x = d1.dot(d1), d1y = d1.dot(d2);
	double d2x = d1y, d2y = d2.dot(d2);
	double d3x = d3.dot(d1), d3y = d3.dot(d2);
	double d = 1.0 / (d1x * d2y - d1y * d2x);
	double l1 = (d2y * d3x - d2x * d3y) * d;
	double l2 = (d1x * d3y - d1y * d3x) * d;
	return Eigen::Vector3d(l1, l2, 1.0 - l1 - l2);
}

// project p to plane(n, v(any point on plane))
inline Eigen::Vector3d project(const Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& v = Eigen::Vector3d::Zero()) {
	return p - (p.dot(n) - v.dot(n)) * n;
}

// distance between p and plane(n, v(any point on plane))
inline double pointPlaneDist(const Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& v = Eigen::Vector3d::Zero()) {
	return p.dot(n) - v.dot(n);
}

/// (triangle edge clamped) signed distance
/// test :
/// std::cout << point2triangleDist(
///   Eigen::Vector3d(0.0, 0.0, 0.0),
///   Eigen::Vector3d(-1.0, 0.0, 0.0),
///   Eigen::Vector3d(1.0, 0.0, 0.0),
///   Eigen::Vector3d(0.0, 1.0, 0.0),
///   Eigen::Vector3d(0.0, 0.0, 1.0));
inline double point2triangleDist(const Eigen::Vector3d& p,
	const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
	const Eigen::Vector3d& n) {

	double dist = p.dot(n) - a.dot(n);

	Eigen::Vector3d proj = p - dist * n;

	Eigen::Vector3d r0 = c - a, r1 = b - a, r2 = proj - a;

	double dot00 = r0.dot(r0);
	double dot01 = r0.dot(r1);
	double dot02 = r0.dot(r2);
	double dot11 = r1.dot(r1);
	double dot12 = r1.dot(r2);

	double denom = (dot00 * dot11 - dot01 * dot01);
	double invDenom = 1.0 / denom;
	double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	if ((u >= 0.) && (v >= 0.) && (u + v <= 1.)) {
		return dist;
	}

	// clamp barycentric coord

	if (u < 0.) {
		double t = std::clamp((proj - b).dot(c - b) / (c - b).dot(c - b), 0., 1.);
		return ((0. <= dist) - (dist < 0.)) * Eigen::Vector3d(
			p[0] - t * c[0] - (1.0 - t) * b[0],
			p[1] - t * c[1] - (1.0 - t) * b[1],
			p[2] - t * c[2] - (1.0 - t) * b[2]).norm();
	} else if (v < 0.) {
		//double t = std::clamp((proj - c).dot(a - c) / (a - c).dot(a - c), 0., 1.);
		double t = std::clamp((proj - c).dot(-r0) / dot00, 0., 1.);
		return ((0. <= dist) - (dist < 0.)) * Eigen::Vector3d(
			p[0] - t * a[0] - (1.0 - t) * c[0],
			p[1] - t * a[1] - (1.0 - t) * c[1],
			p[2] - t * a[2] - (1.0 - t) * c[2]).norm();
	} else {
		//double t = std::clamp((proj - a).dot(b - a) / (b - a).dot(b - a), 0., 1.);
		double t = std::clamp(dot12 / dot11, 0., 1.);
		return ((0. <= dist) - (dist < 0.)) * Eigen::Vector3d(
			p[0] - t * b[0] - (1.0 - t) * a[0],
			p[1] - t * b[1] - (1.0 - t) * a[1],
			p[2] - t * b[2] - (1.0 - t) * a[2]).norm();
	}
}

// linear interpolation
inline Eigen::Vector3d interpolate(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const double t) {
	return Eigen::Vector3d((1.0 - t) * a[0] + b[0] * t,
		(1.0 - t) * a[1] + b[1] * t,
		(1.0 - t) * a[2] + b[2] * t);
}

inline Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
	Eigen::VectorXd ex = x.array().exp();
	return ex.array() / ex.sum();
}

inline Eigen::VectorXd softmin(const Eigen::VectorXd& x) {
	Eigen::VectorXd ex = (-x.array()).exp();
	return ex.array() / ex.sum();
}

// solid angle at p (a, b, c : triangle_point.abc - p)
inline double solidAngle(const Eigen::Vector3d & a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
  double r0 = a.norm(), r1 = b.norm(), r2 = c.norm();
  double d = r0 * r1 * r2 + a.dot(b) * r2 + b.dot(c) * r0 + c.dot(a) * r1;
  return 2. * std::atan2(a.dot(b.cross(c)), d);
}

template <typename Derived, typename T, typename F = typename Derived::Scalar>
Eigen::MatrixX3<F> vertexNormals(const Eigen::MatrixBase<Derived>& pos, const T& faces) {
	size_t n = (size_t)pos.rows();

	Eigen::MatrixX3<F> normals = Eigen::MatrixX3<F>::Zero(n, 3);
	Eigen::Vector3<F> normal;

	if constexpr (IsEigen<T>) {
		for (int i = 0; i < faces.rows(); i++) {
			normal = (pos.row(faces(i, 1)) - pos.row(faces(i, 0))).cross(pos.row(faces(i, 2)) - pos.row(faces(i, 0)));
			normals.row(faces(i, 0)) += normal.transpose();
			normals.row(faces(i, 1)) += normal.transpose();
			normals.row(faces(i, 2)) += normal.transpose();
		}
	} else if constexpr (IsStdvec2d<T>) {
		for (size_t i = 0; i < faces.size(); i++) {
			normal = (pos.row(faces[i][1]) - pos.row(faces[i][0])).cross(pos.row(faces[i][2]) - pos.row(faces[i][0]));
			normals.row(faces[i][0])   += normal.transpose();
			normals.row(faces[i][1]) += normal.transpose();
			normals.row(faces[i][2]) += normal.transpose();
		}
	} else {
		for (size_t i = 0; i < faces.size(); i+=3) {
			normal = (pos.row(faces[i+1]) - pos.row(faces[i])).cross(pos.row(faces[i+2]) - pos.row(faces[i]));
			normals.row(faces[i])   += normal.transpose();
			normals.row(faces[i+1]) += normal.transpose();
			normals.row(faces[i+2]) += normal.transpose();
		}
	}

	normals.rowwise().normalize();
	return normals;
}

template <typename Derived, typename T, typename F = typename Derived::Scalar>
Eigen::MatrixX3<F> faceNormals(const Eigen::MatrixBase<Derived>& pos, const T& faces) {
	Eigen::MatrixX3d normals;

	if constexpr (IsEigen<T>) {
		normals.resize(faces.rows(), 3);
		for (int i = 0; i < faces.rows(); i++) {
			normals.row(i) = (pos.row(faces(i, 1)) - pos.row(faces(i, 0))).cross(pos.row(faces(i, 2)) - pos.row(faces(i, 0)));
		}
	} else if constexpr (IsStdvec2d<T>) {
		normals.resize(faces.size(), 3);
		for (size_t i = 0; i < faces.size(); i++) {
			normals.row(i) = (pos.row(faces[i][1]) - pos.row(faces[i][0])).cross(pos.row(faces[i][2]) - pos.row(faces[i][0]));
		}
	} else {
		normals.resize(faces.size()/3, 3);
		for (size_t i = 0; i < faces.size(); i+=3) {
			normals.row(i/3) = (pos.row(faces[i+1]) - pos.row(faces[i])).cross(pos.row(faces[i+2]) - pos.row(faces[i]));
		}
	}

	return normals;
}

///
/// glm like lookAt function
/// 
inline Eigen::Matrix4d lookAt(const Eigen::Vector3d& eye, const Eigen::Vector3d& center, const Eigen::Vector3d& up) {
	Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();
	Eigen::Vector3d t = eye - center;
	double l = t.norm();
	matrix(2)  = t[0] / l;
	matrix(6)  = t[1] / l;
	matrix(10) = t[2] / l;

	t[0] = up[1] * matrix(10) - up[2] * matrix(6);
	t[1] = up[2] * matrix(2)  - up[0] * matrix(10);
	t[2] = up[0] * matrix(6)  - up[1] * matrix(2);
	l = t.norm();
	matrix(0) = t[0] / l;
	matrix(4) = t[1] / l;
	matrix(8) = t[2] / l;

	matrix(1) = matrix(6) * matrix(8) - matrix(10)* matrix(4);
	matrix(5) = matrix(10) *matrix(0) - matrix(2) * matrix(8);
	matrix(9) = matrix(2) * matrix(4) - matrix(6) * matrix(0);

	matrix(12) = -(eye[0] * matrix(0) + eye[1] * matrix(4) + eye[2] * matrix(8));
	matrix(13) = -(eye[0] * matrix(1) + eye[1] * matrix(5) + eye[2] * matrix(9));
	matrix(14) = -(eye[0] * matrix(2) + eye[1] * matrix(6) + eye[2] * matrix(10));

	matrix(15) = 1.0;
	return matrix;
}


///
/// glm like perspective function
/// 
inline Eigen::Matrix4d perspective(const double fov, const double aspectRatio, const double zNear, const double zFar) {
	double yScale = 1.0 / tan(0.5 * fov);
	double xScale = yScale / aspectRatio;

	Eigen::Matrix4d matrix;
	matrix << xScale, 0, 0, 0,
		0, yScale, 0, 0,
		0, 0, -(zFar+zNear)/(zFar-zNear), -1,
		0, 0, -2*zNear*zFar/(zFar-zNear), 0;

	return matrix;
}

inline Eigen::MatrixXd vertexTexcoords(const Eigen::MatrixXd& pos, const double scale=0.01,
	const Eigen::Vector3d& eye=Eigen::Vector3d::UnitX(), const Eigen::Vector3d& center = Eigen::Vector3d::Zero(),
	const Eigen::Vector3d& up=Eigen::Vector3d::UnitY()) {
	Eigen::Matrix4d model = Eigen::Matrix4d::Identity();
	model(0, 0) = scale;
	model(1, 1) = scale;
	model(2, 2) = scale;
	auto view = lookAt(eye, center, up);
	auto proj = perspective(2.0*0.785398, 1.0, 0.01, 100.0);

	Eigen::Matrix4d mvp = model * view * proj;
	Eigen::MatrixXd uv(pos.rows(), 2);
	for (size_t i = 0; i < (size_t)pos.rows(); i++) {
		Eigen::Vector4d temp = mvp *
			Eigen::Vector4d(pos(i, 0), pos(i, 1), pos(i, 2), 1.0);
		uv(i, 0) = temp(0);
		uv(i, 1) = temp(1);
	}
	return uv;
}

inline Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> heatDiffusionSolver(const Eigen::SparseMatrix<double>& lbo,
	const Eigen::VectorXd& areas, const double time) {
	size_t nv = (size_t)lbo.rows();
	Eigen::SparseMatrix<double> A(nv, nv);
	A.setIdentity();
	A.diagonal().array() *= areas.array();

	return Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>(A + time * lbo);
}

inline Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> laplaceSolver(const Eigen::SparseMatrix<double>& lbo) {
	return Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>(lbo);
}

inline Eigen::VectorXd heatDiffusion(
	const Eigen::SparseMatrix<double>& lbo,
	const Eigen::VectorXd& areas,	const Eigen::VectorXd& func,
	const double time, const size_t step = 1) {
	Eigen::VectorXd ret = func;
	Eigen::SparseMatrix<double> A(lbo.rows(), lbo.rows());
	A.setIdentity();
	A.diagonal().array() *= areas.array();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A + time * lbo);

	for (size_t i = 0; i < step; i++) {
		ret = solver.solve(ret);
	}
	return ret;
}

// TODO
template <typename Derived>
void taubinSmoothing(Eigen::MatrixBase<Derived>& data, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas,
	const size_t steps, const double shrink=0.33, const double inflate=0.34) {
	Eigen::VectorXd invA = 1.0/areas.array();
	Eigen::SparseMatrix<double> S = invA.asDiagonal() * lbo;
	invA = S.diagonal();
	invA = invA.asDiagonal().inverse().diagonal();
	S = invA.asDiagonal() * S;

	for (size_t i = 0; i < steps; i++) {
		data += inflate * S * data;
		data -= shrink * S * data;
	}
}

#endif
}

///
/// colors
/// 
namespace gp::color {
template<typename T, typename S, typename F=T::value_type>
T scalar2rgb(const S scalar) {
	F one = 1.0, zero = 0.0;
	F a = (F)((1.0 - scalar) * 4.0);
	F x;
	F y = modf(a, &x);

	switch ((int)x) {
	case 0:
		return T{ one, y, zero };
	case 1:
		return T{ one - y, one, zero };
	case 2:
		return T{ zero, one, y };
	case 3:
		return T{ zero, one - y, one };
	case 4:
		return T{ zero, zero, one };
	}
	return T{ zero, zero, zero };
}

template<typename T, typename S, typename F = T::value_type>
T scalar2grey(const S scalar) {
	return T{ (F)scalar, (F)scalar, (F)scalar };
}

template<typename T1, typename T2= int, typename F = typename T1::value_type>
T1 randomColor(const T2 seed) {
	T1 color;
	if constexpr (util::IsStdvec<T1>) color.resize(3);
	color[0] = static_cast<F>(cos(((seed ^ (seed << 13)) ^ (seed >> 7)) ^ (seed << 17)) * 0.5 + 0.5);
	color[1] = static_cast<F>(sin(10.0 * std::hash<T2>()(seed)) * 0.5 + 0.5);
	color[2] = static_cast<F>(sin(seed * seed * seed) * 0.5 + 0.5);
	return color;
}

template <typename T, typename S, typename F=T::value_type>
T scalar2rb(const S scalar) {
	F value = (F)2. - static_cast<F>(2 * scalar);
	value *= (F)0.9;
	if (value > (F)0.9) {
		return T{ (F)1.8 - value, (F)1.8 - value, (F)1. };
	}
	return T{ (F)1., value, value};
}

// style : rgb(0), grey(1), rb(2)
template<typename T1, typename T2, typename F = T1::value_type>
std::vector<T1> stepColor(const T2& v, const int style = 0) {
	size_t n = v.size();
	std::vector<size_t> index(n);
	std::iota(index.begin(), index.end(), 0);
	std::sort(index.begin(), index.end(), [&v](auto l, auto r) { return v[l] < v[r]; });

	std::vector<T1> color(n);
	for (size_t i = 0; i < n; i++) {
		if (style == 0)
			color[index[i]] = scalar2rgb<T1>(static_cast<F>(i) / (n-1));
		else if (style == 1)
			color[index[i]] = scalar2grey<T1>(static_cast<F>(i) / (n-1));
		else
			color[index[i]] = scalar2rb<T1>(static_cast<F>(i) / (n-1));
	}
	return color;
}

// style : rgb(0), grey(1), rb(2)
template<typename T1, typename T2, typename F = T2::value_type>
std::vector<T1> normalizedColor(const T2 v, const int style = 0) {
	size_t n = v.size();
	double min = 0.0, max = 1.0;
	std::vector<double> w(v.size());

	if constexpr (util::IsStdvec<T2>) {
		min = *std::min_element(v.begin(), v.end());
		max = *std::max_element(v.begin(), v.end());
	} else if constexpr (util::IsEigen<T2>) {
		min = v.minCoeff();
		max = v.maxCoeff();
	}
	for (size_t i = 0; i < w.size(); i++) w[i] = (v[i] - min) / (max - min);

	std::vector<T1> color(n);
	for (size_t i = 0; i < n; i++) {
		if (style == 0)
			color[i] = scalar2rgb<T1>(static_cast<F>(w[i]));
		else if (style == 1)
			color[i] = scalar2grey<T1>(static_cast<F>(w[i]));
		else
			color[i] = scalar2rb<T1>(static_cast<F>(w[i]));
	}
	return color;
}

// style : rgb(0), grey(1), rb(2)
template<typename T1, typename T2, typename F = typename T1::value_type>
std::vector<T1> indexColor(const T2 n, const int style = 0) {
	std::vector<T1> color(n);
	if constexpr (util::IsStdvec<T1>) color.resize(n, T1(3));
	auto temp = color[0][0];
	for (T2 i = 0; i < n; i++) {
		if (style == 0)
			color[i] = scalar2rgb<T1>(static_cast<F>(i) / (n-1));
		else if (style == 1)
			color[i] = scalar2grey<T1>(static_cast<F>(i) / (n-1));
		else
			color[i] = scalar2rb<T1>(static_cast<F>(i) / (n-1));
	}
	return color;
}

template<typename T, typename V = T::value_type, typename F = V::value_type>
T waveColor(const T color, const int channel = 2, const F c = 16.) {
	size_t n = color.size();
	T wcolor(n);
	if constexpr (util::IsStdvec<V>) wcolor.resize(n, V(3));
	for (size_t i = 0; i < n; i++) {
		wcolor[i][0] = cos(2.0 * std::numbers::pi_v<F> * c * color[i][0]);
		wcolor[i][1] = cos(2.0 * std::numbers::pi_v<F> * c * color[i][1]);
		wcolor[i][2] = cos(2.0 * std::numbers::pi_v<F> * c * color[i][2]);
		wcolor[i][channel] = 1.0;
	}
	return wcolor;
}

template<typename T, typename V = T::value_type, typename F = V::value_type>
T waveColorRB(const T color, float c = 16.0f) {
	std::vector<float> wave(color.size());
	T wcolor(color.size());
	for (size_t i = 0; i < color.size(); i++)
		wave[i] = (float)(2 * pow(cos(2 * std::numbers::pi_v<float> *c * color[i][0]), 3) + 2);

	for (size_t i = 0; i < color.size(); i++) {
		wcolor[i][0] = static_cast<F>(std::exp(-util::pow2(std::max<F>(0.9f, 1.f / (color[i][0] + 0.0001f)) - 0.55f) / 8.f));
		wcolor[i][0] = static_cast<F>(1.f - color[i][0]);
		wcolor[i][1] = static_cast<F>(0.2f);
		wcolor[i][2] = static_cast<F>(std::exp(-util::pow2(std::max<F>(0.9f, 1.f / (color[i][2] + 0.0001f)) - 0.55f) / 8.f));
	}
	for (size_t i = 0; i < color.size(); i++) {
		wcolor[i][0] *= static_cast<F>(1.f);
		wcolor[i][1] *= static_cast<F>(wave[i] > 2.f ? 1.f : wave[i] * 0.5f);
		wcolor[i][2] *= static_cast<F>(1.f);
	}

	return wcolor;
}

}

///
/// misc
/// 
namespace gp::util {
inline void printProgress(double percentage) {
	std::string pb = "■■■■■■■■■■■■■■■■■■■■";
	std::string br = "　　　　　　　　　　　　　　　　　　　　";
	int val = (int)(percentage * 100);
	int l = (int)(percentage * pb.length());
	int r = (int)pb.length() - l;
	std::cout << "\r" << val << "% [" << pb.substr(0, l) << br.substr(0, r) << "]" << std::flush;
}

template <typename Dur = std::chrono::milliseconds>
struct Timer {
	std::chrono::steady_clock::time_point start;
	std::string str;
	bool printStatus;

	Timer(std::string_view str="") : start(std::chrono::high_resolution_clock::now()), str(str), printStatus(true) {}

	void restart() {
		printStatus = true;
		start = std::chrono::high_resolution_clock::now();
	}

	void end() {
		if (printStatus)
			std::cout << str << std::chrono::duration_cast<Dur>(std::chrono::high_resolution_clock::now() - start) << std::endl;
		printStatus = false;
	}

	~Timer() {
		end();
	}
};
}

///
/// global operator overload
/// 
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& pair) {
	out << "(" << pair.first << ", " << pair.second << ")";
	return out;
}

template <typename T, size_t Size>
std::ostream& operator<<(std::ostream& out, const std::array<T, Size>& vec) {
	for (const auto& i : vec) out << i << " ";
	return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
	for (const auto& i : vec) out << i << " ";
	return out;
}

