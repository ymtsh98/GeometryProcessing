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
namespace kika::util {

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

/*
template <typename T>
concept IsScalar = std::is_scalar_v<T>;

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
namespace kika::util {

/// generate (0) to (end-1) vector
template <typename I=size_t>
std::vector<I> vecrange(const size_t end) {
	std::vector<I> vec(end);
	std::iota(vec.begin(), vec.end(), 0);
	return vec;
}

/// generate (start) to (end-1) vector
template <typename I=size_t>
std::vector<I> vecrange(const size_t start, const size_t end) {
	std::vector<I> vec(end - start);
	std::iota(vec.begin(), vec.end(), start);
	return vec;
}

template <typename S1, typename T, typename S2 = typename T::value_type, class = std::void_t<decltype(std::declval<T>().begin())>>
std::vector<S1> vecast(const T& vec) {
	#pragma warning (push) 
	#pragma warning (disable : 4244)
	return std::vector<S1>(vec.begin(), vec.end());
	#pragma warning(pop)
}

template <typename T, typename Idx>
std::vector<T> vecselect(const std::vector<T>& vec, const Idx& select) {
	std::vector<T> ret(select.size());
	for (size_t i = 0; i < ret.size(); i++) ret[i] = vec[select[i]];
	return ret;
}

template <typename T, typename T1, typename T2>
std::vector<T> vecrand(const size_t size, const T1& lower, const T2& higher, const unsigned int seed = std::random_device{}()) {
	std::mt19937 engine(seed);

	std::conditional_t<std::numeric_limits<T>::is_integer,
		std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>
		dis(static_cast<T>(lower), static_cast<T>(higher));
	std::vector<T> v(size);
	for (auto& item : v) item = dis(engine);
	return v;
}

// complement vector of input index vector 
template <typename T = int>
std::vector<T> complement(const size_t size, const std::vector<T>& vec) {
	std::vector<bool> inout(size, false);
	for (auto i : vec) inout[i] = true;

	std::vector<T> covec;
	covec.reserve(size - vec.size());
	for (size_t i = 0; i < size; i++) {
		if (!inout[i]) covec.emplace_back(static_cast<T>(i));
	}
	return covec;
}

template <typename I=size_t>
std::vector<I> randomPermutation(const size_t n, const unsigned long seed = std::random_device{}()) {
	std::mt19937 engine(seed);

	std::vector<I> randomIndices = vecrange<I>(n);
	std::shuffle(randomIndices.begin(), randomIndices.end(), engine);

	return randomIndices;
}

inline double mapping(const double value, const double start1, const double stop1,
	const double start2 = 0.0, const double stop2 = 1.0) {
	return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}

template <typename T1, typename T2>
void wsort(std::vector<T1>& vec, std::vector<T2>& weights) {
	std::vector<uint32_t> idx = vecrange<uint32_t>(vec.size());
	std::sort(idx.begin(), idx.end(), [&weights](uint32_t a, uint32_t b) { return weights[a] < weights[b]; });

	std::vector<T1> rv = vec;
	std::vector<T2> rw = weights;
	std::transform(idx.begin(), idx.end(), rv.begin(), [&vec](uint32_t i) { return vec[i]; });
	std::transform(idx.begin(), idx.end(), rw.begin(), [&weights](uint32_t i) { return weights[i]; });
	vec = rv;
	weights = rw;
}

template <typename T, typename I = uint32_t>
std::vector<I> sortedIota(const T& data) {
	auto range = kika::util::vecrange<I>((size_t)data.size());
	std::sort(range.begin(), range.end(), [&data](auto& l, auto& r) { return data[l] < data[r]; });
	return range;
}

/// operation push_back as increment counter
/// usage : 
/// eg. (size of two vectors union)
/// std::vector<T> s1 = { 1, 2, 3 }, s2 = { 2, 3, 4 }; // or set
/// std::sort(s1.begin(), s1.end());
/// std::sort(s2.begin(), s2.end());
/// pushBackCounter<uint32_t> pbc;
/// std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(pbc));
/// std::cout << pbc.count << std::endl; // return 4
template <typename T>
struct pushBackCounter {
public:
	using value_type = T;

	void push_back(T const&) { ++count; }

	std::size_t count = 0;
};

template <typename T>
size_t numVecIntersect(std::vector<T>&& s1, std::vector<T>&& s2) {
	std::sort(s1.begin(), s1.end());
	std::sort(s2.begin(), s2.end());
	pushBackCounter<uint32_t> pbc;
	std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(pbc));
	return pbc.count;
}

namespace impl {
/// convert col-major vector to row-major
/// 
/// 0, 2, 4, 1, 3, 5 -> 0, 1, 2, 3, 4, 5
/// 
/// 0, 2, 4    0, 1, 2
/// 1, 3, 5 -> 3, 4, 5 
template <typename T, size_t... Seq, size_t Dim = sizeof...(Seq)>
void rowMajorOrder(size_t nv, std::vector<T>& vec, std::index_sequence<Seq...>) {
	auto tmp = vec;
	for (size_t i = 0; i < nv; i++) {
		((tmp[i * Dim + Seq] = vec[i + nv * Seq]), ...);
	}
	vec = tmp;
}

// convert row-major vector to col-major
/// 
/// 0, 1, 2, 3, 4, 5 -> 0, 3, 1, 4, 2, 5
/// 
/// 0, 1, 2    0, 2, 4
/// 3, 4, 5 -> 1, 3, 5
template <typename T, size_t... Seq, size_t Dim = sizeof...(Seq)>
void colMajorOrder(size_t nv, std::vector<T>& vec, std::index_sequence<Seq...>) {
	auto tmp = vec;
	for (size_t i = 0; i < nv; i++) {
		((tmp[i + nv * Seq] = vec[i * Dim + Seq]), ...);
	}
	vec = tmp;
}
}

template <size_t Dim, typename T>
void rowMajorOrder(size_t nv, std::vector<T>& vec) {
	impl::rowMajorOrder(nv, vec, std::make_index_sequence<Dim>());
}


template <size_t Dim, typename T>
void colMajorOrder(size_t nv, std::vector<T>& vec) {
	impl::colMajorOrder(nv, vec, std::make_index_sequence<Dim>());
}

}

///
/// mathematics
/// 
namespace kika::util {
template <typename T>
inline T pow2(const T& a) {
	return a * a;
}

template <typename T>
inline T pow3(const T& a) {
	return a * a * a;
}

template <typename T>
inline T zeroBoundSqrt(const T& a) {
	return (a >= 0) ? sqrt(a) : 0;
}

template <typename T>
T l2norm(const std::vector<T>& vec) {
	T sum = 0.0;
	for (size_t i = 0; i < vec.size(); i++) sum += pow2(vec[i]);
	return sqrt(sum);
}

template <typename T>
inline T sign(const T& a) {
	return (a>0)-(a<0);
}
}

/// geometry processing utility
namespace kika::util {

// cotangent weight (edge(a->b), angle at apex(c))
template<typename T1, typename T2, typename T3>
inline double cot(const T1& a, const T2& b, const T3& c) {
	// double dot = (a[0] - c[0]) * (b[0] - c[0]) +
	// 		(a[1] - c[1]) * (b[1] - c[1]) +
	// 		(a[2] - c[2]) * (b[2] - c[2]);
	// double cross0 = (a[1] - c[1])* (b[2] - c[2]) - (a[2] - c[2]) * (b[1] - c[1]);
	// double cross1 = (a[2] - c[2])* (b[0] - c[0]) - (a[0] - c[0]) * (b[2] - c[2]);
	// double cross2 = (a[0] - c[0])* (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
	// return dot / sqrt(cross0 * cross0 + cross1 * cross1 + cross2 * cross2);

	auto e1 = (a - c).eval();
	auto e2 = (b - c).eval();
	double dot = e1.dot(e2);
	double crossNorm = e1.cross(e2).norm();
	return dot / crossNorm;
}


// @param triangle edge length a, b, c 
inline bool isObtuseTriangle(double a, double b, double c) {
	if (a > b) std::swap(a, b);
	if (b > c) std::swap(b, c);
	return (c * c) > (a * a + b * b);
}

#ifdef EIGEN_MAJOR_VERSION


// unsigned angle
inline double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
	//return atan2(a.cross(b).norm(), a.dot(b));
	return acos(std::min(std::max(a.dot(b) / (a.norm() * b.norm()), -1.0), 1.0));
}

// signed angle (right-handed)
inline double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& n) {
	return atan2(a.cross(b).dot(n), a.dot(b));
	//return sign(a.cross(b).dot(n)) * angle(a, b);
}

// rotate v 
inline Eigen::Vector3d quatRot(const Eigen::Vector3d& v, const Eigen::Vector3d& axis, double angle) {
	angle *= 0.5;
	Eigen::Quaternion q(cos(angle), sin(angle) * axis(0), sin(angle) * axis(1), sin(angle) * axis(2));
	return q * v;
}

// signed volume (triangle mesh)
template <typename T = uint32_t>
inline double computeVolume(const Eigen::MatrixX3d& verts, const std::vector<T>& idx) {
	double volume = 0.;
	for (size_t i = 0; i < idx.size(); i+=3) {
		volume += verts({ idx[i], idx[i+1], idx[i+2] }, Eigen::all).determinant();
	}
	return (1. / 6.) * volume;
}

// distance between p and plane(n, v(any point on plane))
inline double point2planeDist(const Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& v = Eigen::Vector3d::Zero()) {
	return p.dot(n) - v.dot(n);
}

// project p to plane(n, v(any point on plane))
inline Eigen::Vector3d project2plane(const Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& v = Eigen::Vector3d::Zero()) {
	return p - (point2planeDist(p, n, v)) * n;
}

// compute an arbitary tangent basis
inline std::pair<Eigen::Vector3d, Eigen::Vector3d> tangentBasis(const Eigen::Vector3d& n) {
	Eigen::Vector3d v = Eigen::Vector3d::UnitX();

	if (fabs(n.dot(v)) > 0.95) v = Eigen::Vector3d::UnitY();
	Eigen::Vector3d t1 = v.cross(n);
	Eigen::Vector3d t2 = n.cross(t1);
	return std::make_pair(t1, t2);
}

#endif
}

namespace kika::util {
#ifdef EIGEN_MAJOR_VERSION
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

inline Eigen::MatrixX2d texcoordsFromView(const Eigen::MatrixX3d& pos, const double scale=0.01,
	const Eigen::Vector3d& eye=Eigen::Vector3d::UnitX(), 
	const Eigen::Vector3d& center = Eigen::Vector3d::Zero(),
	const Eigen::Vector3d& up=Eigen::Vector3d::UnitY()) {
	Eigen::Matrix4d model = Eigen::Matrix4d::Identity();
	model(0, 0) = scale;
	model(1, 1) = scale;
	model(2, 2) = scale;
	auto view = lookAt(eye, center, up);
	auto proj = perspective(2.0*0.785398, 1.0, 0.01, 100.0);

	Eigen::Matrix4d mvp = model * view * proj;
	Eigen::MatrixX2d uv(pos.rows(), 2);
	for (size_t i = 0; i < (size_t)pos.rows(); i++) {
		Eigen::Vector4d temp = mvp *
			Eigen::Vector4d(pos(i, 0), pos(i, 1), pos(i, 2), 1.0);
		uv(i, 0) = temp(0);
		uv(i, 1) = temp(1);
	}
	return uv;
}
#endif
}

#if false
///
/// colors
/// 
namespace kika::color {
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
	return T{ (F)1., value, value };
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
		wcolor[i][0] = cos(2.0 * std::numbers::pi_v<F> *c * color[i][0]);
		wcolor[i][1] = cos(2.0 * std::numbers::pi_v<F> *c * color[i][1]);
		wcolor[i][2] = cos(2.0 * std::numbers::pi_v<F> *c * color[i][2]);
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
#endif

///
/// misc
/// 
namespace kika::util {
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

	void restart(std::string_view str="") {
		if (str.size()) this->str = str;
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

// filter like operator example
namespace kika::op {

// vec[:] + x
template <typename T>
class add {
	T x;
public:

	add(const T& x) : x(x) {}

	template <typename Scalar>
	auto& operator|(std::vector<Scalar>& v) {
		return v * *this;
	}

	template <typename Scalar>
	friend auto& operator|(std::vector<Scalar>& v, const add& op) {
		auto val = static_cast<Scalar>(op.x);
		for (auto& a : v) a += val;

		return v;
	}
};

// vec[:] * x
template <typename T>
class mul {
	T x;
public:

	mul(const T& x) : x(x) {}

	template <typename Scalar>
	auto& operator|(std::vector<Scalar>& v) {
		return v * *this;
	}

	template <typename Scalar>
	friend auto& operator|(std::vector<Scalar>& v, const mul& op) {
		auto val = static_cast<Scalar>(op.x);
		for (auto& a : v) a *= val;
		return v;
	}
};

// vec[:] = x
template <typename T>
class assign {
	T x;
public:

	assign(const T& x) : x(x) {}

	template <typename Scalar>
	auto& operator|(std::vector<Scalar>& v) {
		return v * *this;
	}

	template <typename Scalar>
	friend auto& operator|(std::vector<Scalar>& v, const assign& op) {
		auto val = static_cast<Scalar>(op.x);
		for (auto& a : v) a = val;
		return v;
	}
};

// vec[vec[:] == x] = y
template <typename T1, typename T2>
class replace {
	T1 x;
	T2 y;
public:
	replace(const T1& x, const T2& y) : x(x), y(y) {}

	template <typename Derived>
	auto& operator|(Eigen::PlainObjectBase<Derived>& v) {
		return v * *this;
	}

	template <typename Derived, typename Scalar = Derived::Scalar>
	friend auto& operator|(Eigen::PlainObjectBase<Derived>& v, const replace& op) {
		std::replace(v.data(), v.data() + v.size(), static_cast<Scalar>(op.x), static_cast<Scalar>(op.y));
		return v;
	}

	template <typename Scalar>
	auto& operator|(std::vector<Scalar>& v) {
		return v * *this;
	}

	template <typename Scalar>
	friend auto& operator|(std::vector<Scalar>& v, const replace& op) {
		std::replace(v.begin(), v.end(), static_cast<Scalar>(op.x), static_cast<Scalar>(op.y));
		return v;
	}

};

template <typename T = uint32_t, class = std::enable_if<std::numeric_limits<T>::is_integer, void>>
class select : public std::vector<T> {
public:
	select() {}
	select(const std::vector<T>& v) : std::vector<T>(v) {}

	template <typename Scalar>
	[[nodiscard]] std::vector<Scalar> operator|(const std::vector<Scalar>& v) {
		return v * *this;
	}

	template <typename Scalar>
	[[nodiscard]] friend std::vector<Scalar> operator|(const std::vector<Scalar>& v, const select& op) {
		std::vector<Scalar> t; t.reserve(op.size());
		for (const auto& a : op) t.emplace_back(v[a]);
		return t;
	}
};

template <typename Idx = uint32_t, typename T = void>
class find {
	T x;
public:

	find(const T& x) : x(x) {}

	template <typename Scalar>
	[[nodiscard]] select<Idx> operator|(const std::vector<Scalar>& v) {
		return v * *this;
	}

	template <typename Scalar>
	[[nodiscard]] friend select<Idx> operator|(const std::vector<Scalar>& v, const find& op) {
		select<Idx> t; t.reserve(v.size());
		auto val = static_cast<Scalar>(op.x);
		for (Idx i = 0; i < (Idx)v.size(); ++i) if (v[i] == val) t.emplace_back(i);
		return t;
	}
};

}

namespace kika::util::impl {
template<typename... Args, size_t... I>
std::string tupleAsString(const std::tuple<Args...>& tuple, std::index_sequence<I...>) {
	std::string s = "(";
	((s += (I == 0 ? "" : ", ") + std::to_string(std::get<I>(tuple))), ...);
	s += ")";
	return s;
}
}

///
/// global operator overload
/// 
namespace {

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& pair) {
	out << "(" << pair.first << ", " << pair.second << ")";
	return out;
}

template<class... Args>
std::ostream& operator<<(std::ostream& out, const std::tuple<Args...>& tuple) {
	out << kika::util::impl::tupleAsString(tuple, std::make_index_sequence<sizeof...(Args)>());
	return  out;
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
}