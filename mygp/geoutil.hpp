#pragma once
#include <algorithm>
#include <numeric>
#include <numbers>
#include <vector>

#if __has_include("glm/glm.hpp")
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#endif

#if __has_include("Eigen/Eigen")
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

#ifdef GLM_ENABLE
#define USE_GLM
#endif

#ifdef EIGEN_CORE_H
#define USE_EIGEN
#endif

namespace gp {
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
concept IsVec = requires(T t) {
	t.push_back;
	t.shrink_to_fit;
};

template <typename T>
concept Is2DVec = requires(T t) {
	IsVec<T>;
	t[0][0];
};

template <typename T>
concept IsScalar = std::is_scalar_v<T>;

}

namespace gp::util {
template<typename T1, typename T2, typename T3>
inline double cot(const T1& a, const T2& b, const T3& c) {
	if constexpr (IsEigen<T1>) {
		#ifdef USE_EIGEN
		auto e1 = b - a;
		auto e2 = c - a;
		double dot = e1.dot(e2);
		double crossNorm = cross3<Eigen::Vector3d>(e1, e2).norm();
		return dot / crossNorm;
		#else
		throw std::runtime_error("Eigen not found");
		#endif
	} else if constexpr (IsGlm<T1>) {
		#ifdef USE_GLM
		auto e1 = b - a;
		auto e2 = c - a;
		double dot = glm::dot(e1, e2);
		double crossNorm = glm::l2Norm(glm::cross(e1, e2));
		return dot / crossNorm;
		#else
		throw std::runtime_error("glm not found");
		#endif
	} else {
		double dot = (b[0] - a[0]) * (c[0] - a[0]) +
			(b[1] - a[1]) * (c[1] - a[1]) +
			(b[2] - a[2]) * (c[2] - a[2]);
		auto cross = cross3<std::vector<double>>(a, b, c);
		return dot / sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
	}
}

template<typename Ret, typename T>
Ret cross3(const T& a, const T& b, const T& c) {
	return { (b[1] - a[1])* (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
						(b[2] - a[2])* (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
						(b[0] - a[0])* (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) };
}

template<typename Ret, typename T1, typename T2>
inline Ret cross3(const T1& a, const T2& b) {
	if constexpr (IsEigen<T1> && IsEigen<Ret>) {
		return Ret(a).cross(Ret(b));
	} else if constexpr (IsGlm<T1> && IsGlm<Ret>) {
		#ifdef USE_GLM
		return glm::cross(a, b);
		#else
		throw std::runtime_error("glm not found");
		#endif
	} else {
		Ret n;
		if constexpr (IsVec<Ret>) n.resize(3);

		n[0] = a[1] * b[2] - a[2] * b[1];
		n[1] = a[2] * b[0] - a[0] * b[2];
		n[2] = a[0] * b[1] - a[1] * b[0];
		return n;
	}
}

template<typename T>
inline T normalized(const T& a) {
	if constexpr (IsEigen<T>) {
		return a.normalized();
	} else if constexpr (IsGlm<T>) {
		#ifdef USE_GLM
		return glm::normalize(a);
		#else
		throw std::runtime_error("glm not found");
		#endif
	} else {
		auto norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
		auto r = a;
		for (size_t i = 0; i < 3; i++) r[i] /= norm;
		return r;
	}
}

inline bool isObtuseTriangle(double a, double b, double c) {
	if (a > b) std::swap(a, b);
	if (b > c) std::swap(b, c);
	if (a > b) std::swap(a, b);

	return (c * c) > (a * a + b * b);
}

inline double mapping(const double value, const double start1, const double stop1,
											const double start2 = 0.0, const double stop2 = 1.0) {
	return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}
}

namespace gp::color {
#ifdef GLM_ENABLE
	template<typename T=glm::vec3, typename S, typename F=double>
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
		}\
		return T{ zero, zero, zero };
	}

	template<typename T1=glm::vec3, typename T2>
	std::vector<T1> stepColor(const T2 v) {
		size_t n = v.size();
		std::vector<size_t> index(n);
		std::iota(index.begin(), index.end(), 0);
		std::sort(index.begin(), index.end(), [&v](auto l, auto r) { return v[l] < v[r]; });

		std::vector<T1> color(n);
		auto temp = color[0][0];
		for (size_t i = 0; i < n; i++) 
			color[index[i]] = scalar2rgb<T1, decltype(temp), decltype(temp)>(static_cast<decltype(temp)>(i) / n);
		return color;
	}

	template<typename T1=glm::vec3, typename T2>
	std::vector<T1> stepColorGrey(const T2 v) {
		size_t n = v.size();
		std::vector<size_t> index(n);
		std::iota(index.begin(), index.end(), 0);
		std::sort(index.begin(), index.end(), [&v](auto l, auto r) { return v[l] < v[r]; });

		std::vector<T1> color(n);
		auto temp = color[0][0];
		for (size_t i = 0; i < v.size(); i++) 
			color[index[i]] = T1(static_cast<decltype(temp)>(i) / n);
		return color;
	}

	template<typename T1=glm::vec3, typename T2>
	std::vector<T1> normalizedColor(const T2 v) {
		size_t n = v.size();
		double min = 0.0, max = 1.0;
		std::vector<double> w(v.size());

		if constexpr (IsVec<T2>) {
			min = std::min(v);
			max = std::max(v);
		} else if constexpr (IsEigen<T2>) {
			min = v.minCoeff();
			max = v.maxCoeff();
		}
		for (size_t i = 0; i < w.size(); i++) w[i] = (v[i] - min) / (max - min);
		
		std::vector<T1> color(n);
		auto temp = color[0][0];
		for (size_t i = 0; i < n; i++) 
			color[i] = scalar2rgb<T1, decltype(temp), decltype(temp)>(static_cast<decltype(temp)>(w[i]));
		return color;
	}

	template<typename T1=glm::vec3, typename T2>
	std::vector<T1> normalizedColorGrey(const T2 v) {
		size_t n = v.size();
		double min = 0.0, max = 1.0;
		std::vector<double> w(v.size());
		if constexpr (IsVec<T2>) {
			min = std::min(v);
			max = std::max(v);
		} else if constexpr (IsEigen<T2>) {
			min = v.minCoeff();
			max = v.maxCoeff();
		}
		for (size_t i = 0; i < w.size(); i++) w[i] = (v[i] - min) / (max - min);
		std::vector<T1> color(n);
		auto temp = color[0][0];
		for (size_t i = 0; i < n; i++) 
			color[i] = T1(static_cast<decltype(temp)>(w[i]));
		return color;
	}

	template<typename T>
	T waveColor(const T color, const int channel = 2, const float c = 16.0f) {
		size_t n = color.size();
		T wcolor(n);
		if constexpr (IsVec<decltype(wcolor[0])>) wcolor.resize(n, decltype(wcolor[0])(3));
		auto temp = color[0][0];
		for (size_t i = 0; i < n; i++) {
			wcolor[i] = cos(2.0f * std::numbers::pi_v<decltype(temp)> *c * color[i]);
			wcolor[i][channel] = 1.0;
		}
		return wcolor;
	}

	template<typename T>
	T waveColorRB(const T color, float c = 16.0f) {
		auto temp = color[0][0];
		std::vector<float> wave(color.size());
		T wcolor(color.size());
		for (size_t i = 0; i < color.size(); i++) 
			wave[i] = (float)(2 * pow(cos(2 * std::numbers::pi_v<float> *c * color[i][0]), 3) + 2);

		for (size_t i = 0; i < color.size(); i++) {
			wcolor[i][0] = static_cast<decltype(temp)>(std::exp(-pow(std::max(0.9f, 1.f / (color[i][0] + 0.0001f)) - 0.55f, 2) / 8.f));
			wcolor[i][0] = static_cast<decltype(temp)>(1.f - color[i][0]);
			wcolor[i][1] = static_cast<decltype(temp)>(0.2f);
			wcolor[i][2] = static_cast<decltype(temp)>(std::exp(-pow(std::max(0.9f, 1.f / (color[i][2] + 0.0001f)) - 0.55f, 2) / 8.f));
		}
		for (size_t i = 0; i < color.size(); i++) {
			wcolor[i][0] *= static_cast<decltype(temp)>(1.f);
			wcolor[i][1] *= static_cast<decltype(temp)>(wave[i] > 2.f ? 1.f : wave[i] * 0.5f);
			wcolor[i][2] *= static_cast<decltype(temp)>(1.f);
		}

		return wcolor;
	}

	template<typename T1=glm::vec3, typename T2, typename F = typename T1::value_type>
	T1 randomColor(const T2 seed) {
		T1 color;
		if constexpr (IsVec<T1>) color.resize(3);
		color[0] = (F)(cos(((seed ^ (seed << 13)) ^ (seed >> 7)) ^ (seed << 17)) * 0.5 + 0.5);
		color[1] = (F)(sin(10.0 * std::hash<T2>()(seed)) * 0.5 + 0.5);
		color[2] = (F)(sin(seed * seed * seed) * 0.5 + 0.5);
		return color;
	}

	template<typename T1=glm::vec3, typename T2>
	std::vector<T1> indexColor(const T2 n) {
		std::vector<T1> color(n);
		if constexpr (IsVec<decltype(color[0])>) color.resize(n, T1(3));
		auto temp = color[0][0];
		for (T2 i = 0; i < n; i++) {
			color[i] = scalar2rgb<T1, decltype(temp), decltype(temp)>(static_cast<decltype(temp)>(i) / n);
		}
		return color;
	}
#endif
}

namespace gp::util {
void printProgress(double percentage) {
	std::string pb = "■■■■■■■■■■■■■■■■■■■■";
	std::string br = "　　　　　　　　　　　　　　　　　　　　";
	int val = (int)(percentage * 100);
	int l = (int)(percentage * pb.length());
	int r = (int)pb.length() - l;
	std::cout << "\r" << val << "% [" << pb.substr(0, l) << br.substr(0, r) << "]" << std::flush;
}
}