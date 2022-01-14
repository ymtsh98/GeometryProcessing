#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <numbers>
#include <vector>
#include <iostream>
#include <queue>

#if __has_include("glm/glm.hpp")
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_operation.hpp>
#define GLM_INCLUDE
#endif

#if __has_include("Eigen/Eigen")
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Sparse"
#define EIGEN_INCLUDE
#endif

namespace gp {
	template <typename T>
	concept checkStdVec = requires(T t) {
		t.push_back;
		t.shrink_to_fit;
	};

	template <typename T>
	concept checkGlmVec = requires(T t) {
		t.x;
		t.r;
		t.s;
	};

	template <typename T>
	concept checkEigenVec = requires(T t) {
		t.Identity;
		t.col;
	};

	std::vector<size_t> iota(size_t n) {
		std::vector<size_t> index(n);
		std::iota(index.begin(), index.end(), 0);
		return index;
	}

	template <typename T1, typename T2, typename T3>
	double mapping(const T1 value, const T2 start1, const T3 stop1,
								 const double start2 = 0.0, const double stop2 = 1.0) {
		return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
	}

	template <typename T1, typename T2>
	void addVector(std::vector<T1>& a, std::vector<T2>& b) {
		std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T1>());
	}

	template <typename T>
	T faceNormal(const T& a, const T& b, const T& c) {
		if constexpr (checkGlmVec<T>) {
			#ifndef GLM_INCLUDE
			return {};
			#else
			return glm::cross(b - a, c - a);
			#endif
		} else if constexpr (checkEigenVec<T>) {
			return (b - a).cross(c - a);
		} else {
			return { 
				(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
				(b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
				(b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
			};
		}
	}

	template <typename T>
	void normalize(T& a) {
		if constexpr (checkGlmVec<T>) {
			#ifdef GLM_INCLUDE
			a = glm::normalize(a);
			#endif
		} else if constexpr (checkEigenVec<T>) {
			a.normalize();
		} else {
			double s = 0.0;
			for (auto& ai : a) s += ai * ai;
			s = sqrt(s);
			std::for_each(a.begin(), a.end(), [s](auto& ai) { ai /= s; });
		}
	}

	template <typename T1, typename T2>
	T1 calcVertexNormal(const T1& pos, const T2& indices) {
		auto p = pos[0];

		T1 normal;
		if constexpr (checkStdVec<decltype(p)>)
			normal.resize(pos.size(), decltype(p)(p.size(), 0));
		else 
			normal.resize(pos.size(), decltype(p)(0));

		for (size_t i = 0; i < indices.size(); i += 3) {
			auto n = faceNormal(pos[indices[i]], pos[indices[i + 1]], pos[indices[i + 2]]);
			if constexpr (checkStdVec<decltype(p)>) {
				addVector(normal[indices[i]], n);
				addVector(normal[indices[i + 1]], n);
				addVector(normal[indices[i + 2]], n);
			} else if constexpr(checkEigenVec<decltype(p)>) {
				normal[indices[i]] += n;
				normal[indices[i + 1]] += n;
				normal[indices[i + 2]] += n;
			} else {
				normal[indices[i]] += n;
				normal[indices[i + 1]] += n;
				normal[indices[i + 2]] += n;
			}
		}

		for (auto& v : normal) normalize(v);
		return normal;
	}

	namespace color {
		template<typename T, typename C>
		T scalar2rgb(const C scalar) {
			float a = (1.0f - scalar) * 4.0;
			float x;
			float y = modf(a, &x);

			T color = { 0.0f, 0.0f, 0.0f };

			switch ((int)x) {
			case 0:
				color = { 1.0f, y, 0.0f };
				break;
			case 1:
				color = { 1.0f - y, 1.0f, 0.0f }; break;
			case 2:
				color = { 0.0f, 1.0f, y }; break;
			case 3:
				color = { 0.0f, 1.0f - y, 1.0f }; break;
			case 4:
				color = { 0.0f, 0.0f, 1.0f }; break;
			}

			return color;
		}

		template<typename T1, typename T2>
		std::vector<T1> stepColor(T2 v) {
			size_t n = v.size();
			std::vector<size_t> index(n);
			std::iota(index.begin(), index.end(), 0);
			std::sort(index.begin(), index.end(), [&v](auto l, auto r) { return v[l] < v[r]; });

			std::vector<T1> color(n);
			for (size_t i = 0; i < v.size(); i++) color[index[i]] = scalar2rgb<T1>((float)i / n);
			return color;
		}

		template<typename T1, typename T2>
		std::vector<T1> stepColorGrey(T2 v) {
			size_t n = v.size();
			std::vector<size_t> index(n);
			std::iota(index.begin(), index.end(), 0);
			std::sort(index.begin(), index.end(), [&v](auto l, auto r) { return v[l] < v[r]; });

			std::vector<T1> color(n);
			for (size_t i = 0; i < v.size(); i++) color[index[i]] = T1((float)i / n);
			return color;
		}

		template<typename T1, typename T2>
		std::vector<T1> normalizedColor(T2 v) {
			size_t n = v.size();
			double min = 0.0, max = 1.0;
			if constexpr (checkStdVec<T2>) {
				min = std::min(v);
				max = std::max(v);
				std::for_each(v.begin(), v.end(), [min, max](auto& vi) { vi = (vi - min) / (max - min); });
			} else if constexpr (checkEigenVec<T2>) {
				min = v.minCoeff();
				max = v.maxCoeff();
				v.array() -= min;
				v.array() /= (max - min);
			}
			std::vector<T1> color(n);
			for (size_t i = 0; i < v.size(); i++) color[i] = scalar2rgb<T1>(v[i]);
			return color;
		}

		template<typename T1, typename T2>
		std::vector<T1> normalizedColorGrey(T2 v) {
			size_t n = v.size();
			double min = 0.0, max = 1.0;
			if constexpr (checkStdVec<T2>) {
				min = std::min(v);
				max = std::max(v);
				std::for_each(v.begin(), v.end(), [min, max](auto& vi) { vi = (vi - min) / (max - min); });
			} else if constexpr (checkEigenVec<T2>) {
				min = v.minCoeff();
				max = v.maxCoeff();
				v.array() -= min;
				v.array() /= (max - min);
			}
			std::vector<T1> color(n);
			for (size_t i = 0; i < v.size(); i++) color[i] = T1(v[i]);
			return color;
		}

		template<typename T>
		T waveColor(T color, float c = 16.0f) {
			T wcolor(color.size());
			for (size_t i = 0; i < color.size(); i++) {
				wcolor[i] = cos(2.0f * std::numbers::pi_v<float> * c * color[i]);
				wcolor[i][2] = 1.0;
			}
			return wcolor;
		}

		template<typename T>
		glm::vec3 randomColor(T seed) {
			glm::vec3 color;
			color[0] = cos(((seed ^ (seed << 13)) ^ (seed >> 7)) ^ (seed << 17)) * 0.5 + 0.5;
			color[1] = sin(10.0 * std::hash<T>()(seed)) * 0.5 + 0.5;
			color[2] = sin(seed * seed * seed) * 0.5 + 0.5;
			return color;
		}
	}

	template<typename T>
	bool boxIntersectsSphere(T min_corner, T max_corner, T center, float r) {
		float r2 = r * r;
		float dmin = 0;
		for (size_t i = 0; i < 3; i++) {
			if (center[i] < min_corner[i]) dmin += pow(center[i] - min_corner[i], 2);
			else if (center[i] > max_corner[i]) dmin += pow(center[i] - max_corner[i], 2);
		}
		return dmin <= r2;
	}

	#ifdef GLM_INCLUDE
	template <typename T1, typename T2>
	std::vector<T1> calcVertexTexcoord(const T2& pos, float scale, glm::vec3 s = { -1, 0, 0 }, glm::vec3 t = { 1, 0, 0 }) {
		std::vector<T1> texcoord(pos.size());
		
		glm::mat4 model = glm::diagonal4x4(glm::vec4(glm::vec3(scale), 1.0f));
		glm::mat4 view = glm::lookAt(s, t, glm::vec3(0.0, 1.0, 0.0));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f),1.0f, 0.01f, 100.0f);
		glm::mat4 mvp = proj * view * model;
		for (size_t i = 0; i < pos.size(); i++) {
			auto tp = mvp * glm::vec4(pos[i][0], pos[i][1],	pos[i][2], 1);
			texcoord[i] = { tp[0], tp[1] };
		}
		return texcoord;
	}
	#endif

};