#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <string>

namespace kika::color {

class RGBA;
class fRGBA;

class RGBA {
	std::array<uint8_t, 4> rgba;

public:
	RGBA() : rgba({ 255u, 255u, 255u, 255u }) {}

	RGBA(const uint32_t rgbaHexFormat) {
		rgba[3] = (rgbaHexFormat >> 0u) & 0xffu;
		rgba[2] = (rgbaHexFormat >> 8u) & 0xffu;
		rgba[1] = (rgbaHexFormat >> 16u) & 0xffu;
		rgba[0] = (rgbaHexFormat >> 24u) & 0xffu;
	}

	RGBA(const uint8_t r, const uint8_t g, const uint8_t b, const uint8_t a = 255u) : rgba({ r, g, b, a }) {}

	RGBA(const kika::color::fRGBA& color);

	// -------------------- element access operators --------------------
	uint8_t& r() {
		return rgba[0];
	}

	uint8_t r() const {
		return rgba[0];
	}

	uint8_t& g() {
		return rgba[1];
	}

	uint8_t g() const {
		return rgba[1];
	}

	uint8_t& b() {
		return rgba[2];
	}

	uint8_t b() const {
		return rgba[2];
	}

	uint8_t& a() {
		return rgba[3];
	}

	uint8_t a() const {
		return rgba[3];
	}

	uint8_t& operator[](int i) {
		return rgba[i];
	}

	uint8_t operator[](int i) const {
		return rgba[i];
	}

	// -------------------- operator overload --------------------

	RGBA& operator+=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] = (uint8_t)std::clamp((uint32_t)rgba[i] + c.rgba[i], 0u, 255u);
		return *this;
	}

	RGBA operator+(const RGBA& c) const {
		RGBA ret = *this;
		return ret += c;
	}

	RGBA& operator-=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] = rgba[i] < c.rgba[i] ? 0 : rgba[i] - c.rgba[i];
		return *this;
	}

	RGBA operator-(const RGBA& c) const {
		RGBA ret = *this;
		return ret -= c;
	}

	RGBA& operator*=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] = (uint8_t)std::clamp((uint32_t)rgba[i] * c.rgba[i], 0u, 255u);
		return *this;
	}

	RGBA operator*(const RGBA& c) const {
		RGBA ret = *this;
		return ret *= c;
	}

	RGBA& operator/=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] /= c.rgba[i];
		return *this;
	}

	RGBA operator/(const RGBA& c) const {
		RGBA ret = *this;
		return ret /= c;
	}

	RGBA& operator%=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] %= c.rgba[i];
		return *this;
	}

	RGBA operator%(const RGBA& c) const {
		RGBA ret = *this;
		return ret %= c;
	}

	RGBA& operator<<=(const size_t n) {
		for (int i = 0; i < 4; i++)
			rgba[i] <<= n;
		return *this;
	}

	RGBA operator<<(const size_t n) const {
		RGBA ret = *this;
		return ret <<= n;
	}

	RGBA& operator>>=(const size_t n) {
		for (int i = 0; i < 4; i++)
			rgba[i] >>= n;
		return *this;
	}

	RGBA operator>>(const size_t n) const {
		RGBA ret = *this;
		return ret >>= n;
	}

	RGBA& operator&=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] &= c.rgba[i];
		return *this;
	}

	RGBA operator&(const RGBA& c) const {
		RGBA ret = *this;
		return ret &= c;
	}

	RGBA& operator|=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] |= c.rgba[i];
		return *this;
	}

	RGBA operator|(const RGBA& c) const {
		RGBA ret = *this;
		return ret |= c;
	}

	RGBA& operator^=(const RGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] ^= c.rgba[i];
		return *this;
	}

	RGBA operator^(const RGBA& c) const {
		RGBA ret = *this;
		return ret ^= c;
	}

	RGBA operator<(const RGBA& c) const {
		return RGBA(
			(rgba[0] < c.rgba[0]) * 255u,
			(rgba[1] < c.rgba[1]) * 255u,
			(rgba[2] < c.rgba[2]) * 255u,
			(rgba[3] < c.rgba[3]) * 255u);
	}

	RGBA operator<=(const RGBA& c) const {
		return RGBA(
			(rgba[0] <= c.rgba[0]) * 255u,
			(rgba[1] <= c.rgba[1]) * 255u,
			(rgba[2] <= c.rgba[2]) * 255u,
			(rgba[3] <= c.rgba[3]) * 255u);
	}

	RGBA operator>(const RGBA& c) const {
		return RGBA(
			(rgba[0] > c.rgba[0]) * 255u,
			(rgba[1] > c.rgba[1]) * 255u,
			(rgba[2] > c.rgba[2]) * 255u,
			(rgba[3] > c.rgba[3]) * 255u);
	}

	RGBA operator>=(const RGBA& c) const {
		return RGBA(
			(rgba[0] >= c.rgba[0]) * 255u,
			(rgba[1] >= c.rgba[1]) * 255u,
			(rgba[2] >= c.rgba[2]) * 255u,
			(rgba[3] >= c.rgba[3]) * 255u);
	}

	RGBA operator==(const RGBA& c) const {
		return RGBA(
			(rgba[0] == c.rgba[0]) * 255u,
			(rgba[1] == c.rgba[1]) * 255u,
			(rgba[2] == c.rgba[2]) * 255u,
			(rgba[3] == c.rgba[3]) * 255u);
	}

	RGBA operator!=(const RGBA& c) const {
		return RGBA(
			(rgba[0] != c.rgba[0]) * 255u,
			(rgba[1] != c.rgba[1]) * 255u,
			(rgba[2] != c.rgba[2]) * 255u,
			(rgba[3] != c.rgba[3]) * 255u);
	}

	RGBA operator&&(const RGBA& c) const {
		return RGBA(
			(rgba[0] && c.rgba[0]) * 255u,
			(rgba[1] && c.rgba[1]) * 255u,
			(rgba[2] && c.rgba[2]) * 255u,
			(rgba[3] && c.rgba[3]) * 255u);
	}

	RGBA operator||(const RGBA& c) const {
		return RGBA(
			(rgba[0] || c.rgba[0]) * 255u,
			(rgba[1] || c.rgba[1]) * 255u,
			(rgba[2] || c.rgba[2]) * 255u,
			(rgba[3] || c.rgba[3]) * 255u);
	}

	// -------------------- utility --------------------
public:

	bool any() const {
		return (rgba[0] > 0u) || (rgba[1] > 0u) || (rgba[2] > 0u) || (rgba[3] > 0u);
	}

	bool all() const {
		return (rgba[0] > 0u) && (rgba[1] > 0u) && (rgba[2] > 0u) && (rgba[3] > 0u);
	}

	RGBA mix(const RGBA& c, const double t = 0.5) const {
		return RGBA(
			(uint8_t)std::clamp(rgba[0] * (1.0 - t) + c.rgba[0] * t, 0., 255.),
			(uint8_t)std::clamp(rgba[1] * (1.0 - t) + c.rgba[1] * t, 0., 255.),
			(uint8_t)std::clamp(rgba[2] * (1.0 - t) + c.rgba[2] * t, 0., 255.),
			(uint8_t)std::clamp(rgba[3] * (1.0 - t) + c.rgba[3] * t, 0., 255.)
		);
	}

	RGBA gammaCorrection() const {
		return RGBA(
			(uint8_t)std::clamp((pow(rgba[0] / 255., 2.2) * 255.), 0., 255.),
			(uint8_t)std::clamp((pow(rgba[1] / 255., 2.2) * 255.), 0., 255.),
			(uint8_t)std::clamp((pow(rgba[2] / 255., 2.2) * 255.), 0., 255.),
			rgba[3]
		);
	}

	RGBA grayscale() const {
		uint8_t c = (uint8_t)std::clamp(0.3 * r() + 0.59 * g() + 0.11 * b(), 0., 255.);
		return RGBA(c, c, c, rgba[3]);
	}
};

class fRGBA {
	std::array<double, 4> rgba;

public:
	fRGBA() : rgba({ 0., 0., 0., 1. }) {}

	fRGBA(const double v) : rgba({ v, v, v, v }) {}

	fRGBA(const double r, const double g, const double b, const double a = 1.) : rgba({ r, g, b, a }) {}

	fRGBA(const kika::color::RGBA& color) {
		this->r() = color.r() / 255.;
		this->g() = color.g() / 255.;
		this->b() = color.b() / 255.;
		this->a() = color.a() / 255.;
	}

	// -------------------- element access operators --------------------
	double& r() {
		return rgba[0];
	}

	double r() const {
		return rgba[0];
	}

	double& g() {
		return rgba[1];
	}

	double g() const {
		return rgba[1];
	}

	double& b() {
		return rgba[2];
	}

	double b() const {
		return rgba[2];
	}

	double& a() {
		return rgba[3];
	}

	double a() const {
		return rgba[3];
	}

	double& operator[](int i) {
		return rgba[i];
	}

	double operator[](int i) const {
		return rgba[i];
	}

	// -------------------- operator overload --------------------

	fRGBA& operator+=(const fRGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] = std::clamp(rgba[i] + c.rgba[i], 0., 1.);
		return *this;
	}

	fRGBA operator+(const fRGBA& c) const {
		fRGBA ret = *this;
		return ret += c;
	}

	fRGBA& operator-=(const fRGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] = rgba[i] < c.rgba[i] ? 0 : rgba[i] - c.rgba[i];
		return *this;
	}

	fRGBA operator-(const fRGBA& c) const {
		fRGBA ret = *this;
		return ret -= c;
	}

	fRGBA& operator*=(const fRGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] = std::clamp(rgba[i] * c.rgba[i], 0., 1.);
		return *this;
	}

	fRGBA operator*(const fRGBA& c) const {
		fRGBA ret = *this;
		return ret *= c;
	}

	fRGBA& operator/=(const fRGBA& c) {
		for (int i = 0; i < 4; i++)
			rgba[i] /= c.rgba[i];
		return *this;
	}

	fRGBA operator/(const fRGBA& c) const {
		fRGBA ret = *this;
		return ret /= c;
	}

	fRGBA operator<(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] < c.rgba[0]), (rgba[1] < c.rgba[1]),
			(rgba[2] < c.rgba[2]), (rgba[3] < c.rgba[3]));
	}

	fRGBA operator<=(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] <= c.rgba[0]), (rgba[1] <= c.rgba[1]),
			(rgba[2] <= c.rgba[2]), (rgba[3] <= c.rgba[3]));
	}

	fRGBA operator>(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] > c.rgba[0]), (rgba[1] > c.rgba[1]),
			(rgba[2] > c.rgba[2]), (rgba[3] > c.rgba[3]));
	}

	fRGBA operator>=(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] >= c.rgba[0]), (rgba[1] >= c.rgba[1]),
			(rgba[2] >= c.rgba[2]), (rgba[3] >= c.rgba[3]));
	}

	fRGBA operator==(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] == c.rgba[0]), (rgba[1] == c.rgba[1]),
			(rgba[2] == c.rgba[2]), (rgba[3] == c.rgba[3]));
	}

	fRGBA operator!=(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] != c.rgba[0]), (rgba[1] != c.rgba[1]),
			(rgba[2] != c.rgba[2]), (rgba[3] != c.rgba[3]));
	}

	fRGBA operator&&(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] && c.rgba[0]), (rgba[1] && c.rgba[1]),
			(rgba[2] && c.rgba[2]), (rgba[3] && c.rgba[3]));
	}

	fRGBA operator||(const fRGBA& c) const {
		return fRGBA(
			(rgba[0] || c.rgba[0]), (rgba[1] || c.rgba[1]),
			(rgba[2] || c.rgba[2]), (rgba[3] || c.rgba[3]));
	}

	// -------------------- utility --------------------
public:

	bool any() const {
		return (rgba[0] > 0u) || (rgba[1] > 0u) || (rgba[2] > 0u) || (rgba[3] > 0u);
	}

	bool all() const {
		return (rgba[0] > 0u) && (rgba[1] > 0u) && (rgba[2] > 0u) && (rgba[3] > 0u);
	}

	fRGBA mix(const fRGBA& c, const double t = 0.5) const {
		return fRGBA(
			std::clamp(rgba[0] * (1.0 - t) + c.rgba[0] * t, 0., 1.),
			std::clamp(rgba[1] * (1.0 - t) + c.rgba[1] * t, 0., 1.),
			std::clamp(rgba[2] * (1.0 - t) + c.rgba[2] * t, 0., 1.),
			std::clamp(rgba[3] * (1.0 - t) + c.rgba[3] * t, 0., 1.)
		);
	}

	fRGBA gammaCorrection() const {
		return fRGBA(
			pow(rgba[0], 2.2), pow(rgba[1], 2.2),
			pow(rgba[2], 2.2), rgba[3]
		);
	}

	fRGBA grayscale() const {
		double c = std::clamp(0.3 * r() + 0.59 * g() + 0.11 * b(), 0., 1.);
		return fRGBA(c, c, c, rgba[3]);
	}
};

RGBA::RGBA(const kika::color::fRGBA& color) {
	r() = (uint8_t)std::clamp(color.r() * 255., 0., 255.);
	g() = (uint8_t)std::clamp(color.g() * 255., 0., 255.);
	b() = (uint8_t)std::clamp(color.b() * 255., 0., 255.);
	a() = (uint8_t)std::clamp(color.a() * 255., 0., 255.);
}

class ColorRamp {
	// todo : flat-map
	std::vector<std::pair<RGBA, double>> colors;

	double mapping(const double value, const double start1, const double end1,
		const double start2 = 0.0, const double end2 = 1.0) const {
		return start2 + (end2 - start2) * ((value - start1) / (end1 - start1));
	}

	void sort() {
		std::sort(colors.begin(), colors.end(), [](auto& l, auto& r) {return l.second < r.second; });
	}

public:

	ColorRamp(const std::vector<RGBA>& colors, const std::vector<double>& weights) {
		assert((colors.size() >= 2) && "Initialization of ColorRamp");
		assert((colors.size() == weights.size()) && "Initialization of ColorRamp");
		size_t n = colors.size();
		for (size_t i = 0; i < n; i++) {
			this->colors.emplace_back(colors[i], weights[i]);
		}
		sort();
	}

	ColorRamp(const std::vector<RGBA>& colors) {
		assert((colors.size() >= 2) && "Initialization of ColorRamp");
		size_t n = colors.size();
		for (size_t i = 0; i < n; i++) {
			this->colors.emplace_back(colors[i], (double)i / (n - 1));
		}
	}

	ColorRamp() {
		colors.emplace_back(RGBA(0, 0, 0, 255), 0.);
		colors.emplace_back(RGBA(255, 255, 255, 255), 1.);
	}

	void append(const RGBA& color, double weight) {
		weight = (weight < 0. ? 0. : (weight > 1. ? 1. : weight));
		colors.emplace_back(color, weight);
		sort();
	}

	void set(size_t i, double weight) {
		weight = (weight < 0. ? 0. : (weight > 1. ? 1. : weight));
		colors[i].second = weight;
		sort();
	}

	void set(size_t i, const RGBA& color) {
		colors[i].first = color;
	}

	void set(size_t i, const RGBA& color, double weight) {
		set(i, color);
		set(i, weight);
	}

	RGBA operator()(double t) const {
		t = (t < 0. ? 0. : (t > 1. ? 1. : t));
		// get right value and decrement after copy
		auto left = std::upper_bound(colors.begin(), colors.end(), t, [](double t, auto& val) { return t < val.second; });
		if (left == colors.begin()) return left->first;
		auto right = left--;
		if (right == colors.end()) return left->first;

		t = mapping(t, left->second, right->second);
		return left->first.mix(right->first, t);
	}

	std::pair<RGBA, double> operator[](size_t i) const {
		return colors[i];
	}
};

/// -------------------- color palette --------------------
namespace palette {
inline ColorRamp rainbow() {
	return ColorRamp{ {{ 0, 0, 255}, {0, 255, 0}, {255, 0, 0}} };
}

inline ColorRamp blueRed() {
	return ColorRamp{ { 0x1012eeffu, 0x6686ffffu, 0xf3efe0ffu, 0xdb3a34ffu, 0xee1210ffu} };
	// return ColorRamp{ { 0x1012eeffu, 0xf3efe0ffu, 0xee1210ffu} };
}

/// -------------------- https://www.kennethmoreland.com/color-advice/  --------------------

inline ColorRamp coolWarm() {
	return ColorRamp{ {{ 59, 78, 192}, { 242, 242, 242 }, { 180, 4, 38}} }; // bent
	// return ColorRamp{ {{ 59, 76, 192}, { 221, 221, 221 }, { 180, 4, 38}} }; // smooth
}

inline ColorRamp viridis() {
	return ColorRamp{
		{	{ 68, 1, 84 }, { 70, 50, 127}, { 54, 92, 141},
			{ 39, 127, 142}, { 31, 161, 135}, { 74, 194, 109},
			{159, 218, 58}, { 253, 231, 37}	},
	};
}

inline ColorRamp plasma() {
	return ColorRamp{
		{	{ 13, 8, 135 }, { 84, 2, 163}, { 139, 10, 165},
			{ 185, 50, 137}, { 219, 92, 104}, { 244, 136, 73},
			{ 254, 188, 43}, { 240, 249, 33 }	},
	};
}

inline ColorRamp blackBody() {
	return ColorRamp{
		{	{ 0, 0, 0 }, { 178, 34, 34 }, { 227, 105, 5},
			{ 230, 230, 53}, { 255, 255, 255 } },
		{ 0.0, 0.39, 0.58, 0.89, 1.}
	};
}

inline ColorRamp inferno() {
	return ColorRamp{
		{	{ 0, 0, 4 }, { 40, 11, 84}, { 101, 21, 110},
			{ 159, 42, 99}, { 212, 72, 66}, { 245, 125, 21},
			{ 250, 193, 39}, { 252, 255, 164 }	},
	};
}

}

// -------------------- utility --------------------

template <typename T>
std::vector<std::array<double, 3>> scalar2color(const T& values, const ColorRamp& plt = palette::blueRed()) {
	auto copy = values;
	double min = *std::min_element(copy.begin(), copy.end());
	for (auto& v : copy) v -= min;
	double max = *std::max_element(copy.begin(), copy.end());
	for (auto& v : copy) v /= max;
	std::vector<std::array<double, 3>> colors(copy.size());
	for (size_t i = 0; i < colors.size(); i++) {
		fRGBA rgba = plt(copy[i]);
		colors[i][0] = rgba.r();
		colors[i][1] = rgba.g();
		colors[i][2] = rgba.b();
	}
	return colors;
}

inline std::vector<std::array<double, 3>> gradation(const size_t n, const ColorRamp& plt = palette::blueRed()) {
	std::vector<std::array<double, 3>> colors(n);
	for (size_t i = 0; i < colors.size(); i++) {
		fRGBA rgba = plt((double)i / (n - 1));
		colors[i][0] = rgba.r();
		colors[i][1] = rgba.g();
		colors[i][2] = rgba.b();
	}
	return colors;
}

}

namespace {
std::ostream& operator<<(std::ostream& out, const kika::color::RGBA& c) {
	out << "(" + std::to_string(c.r()) + "," + std::to_string(c.g()) + "," + std::to_string(c.b()) + "," + std::to_string(c.a()) + ")";
	return out;
}
}

/// old version
/*
namespace kika::color::impl {

template <typename T, class = void>
struct IsColorScalar : std::false_type {};

template <typename T>
struct IsColorScalar<T, std::void_t<decltype(std::declval<T>().scalar)>> : std::true_type {};

template <typename T>
inline uint8_t scalar2u8(const T& scalar) {
	if constexpr (std::is_scalar_v<T>) {
		return static_cast<uint8_t>(std::max(std::min((uint32_t)scalar, 255u), 0u));
	} else if constexpr (IsColorScalar<T>::value) {
		return static_cast<uint8_t>(std::max(std::min((uint32_t)scalar.scalar, 255u), 0u));
	} else {
		static_assert(false && "scalar2u8");
	}
}

template <class Derived>
struct RGBBasicOp {
protected:
	template <typename T>
	static auto scalar2u32float(const T& scalar) {
		if constexpr (std::is_floating_point_v<T>) {
			return scalar;
		} else if constexpr (std::is_scalar_v<T>) {
			return static_cast<uint32_t>(scalar);
		} else if constexpr (IsColorScalar<T>::value) {
			return static_cast<uint32_t>(scalar.scalar);
		} else {
			static_assert(false && "scalar2u32float");
		}
	}

public:
	template <typename T>
	friend Derived operator+(const Derived& d, const T& scalar) {
		return Derived(d.scalar + scalar2u32float(scalar));
	}

	template <typename T>
	friend Derived operator-(const Derived& d, const T& scalar) {
		return Derived(d.scalar - scalar2u32float(scalar));
	}

	template <typename T>
	friend Derived operator*(const Derived& d, const T& scalar) {
		return Derived(d.scalar * scalar2u32float(scalar));
	}

	template <typename T>
	friend Derived operator/(const Derived& d, const T& scalar) {
		return Derived(d.scalar / scalar2u32float(scalar));
	}

	template <typename T>
	friend Derived operator%(const Derived& d, const T& scalar) {
		return Derived(d.scalar % scalar2u32float(scalar));
	}

	template <typename T>
	friend Derived& operator+=(Derived& d, const T& scalar) {
		d.scalar += scalar2u32float(scalar);
		return d;
	}

	template <typename T>
	friend Derived& operator-=(Derived& d, const T& scalar) {
		d.scalar -= scalar2u32float(scalar);
		return d;
	}

	template <typename T>
	friend Derived& operator*=(Derived& d, const T& scalar) {
		d.scalar *= scalar2u32float(scalar);
		return d;
	}

	template <typename T>
	friend Derived& operator/=(Derived& d, const T& scalar) {
		d.scalar /= scalar2u32float(scalar);
		return d;
	}

	template <typename T>
	friend Derived& operator%=(Derived& d, const T& scalar) {
		d.scalar %= scalar2u32float(scalar);
		return d;
	}

	friend Derived operator<<(const Derived& d, const size_t n) {
		return Derived(d.scalar <<= n);
	}

	friend Derived operator>>(const Derived& d, const size_t n) {
		return Derived(d.scalar >>= n);
	}

	friend Derived& operator<<=(Derived& d, const size_t n) {
		d.scalar <<= n;
		return d;
	}

	friend Derived& operator>>=(Derived& d, const size_t n) {
		d.scalar >>= n;
		return d;
	}

	template <typename T>
	friend Derived operator&(const Derived& d, const T& scalar) {
		return Derived(d.scalar & scalar2u8(scalar));
	}

	template <typename T>
	friend Derived operator^(const Derived& d, const T& scalar) {
		return Derived(d.scalar ^ scalar2u8(scalar));
	}

	template <typename T>
	friend Derived operator|(const Derived& d, const T& scalar) {
		return Derived(d.scalar | scalar2u8(scalar));
	}

	template <typename T>
	friend Derived& operator&=(Derived& d, const T& scalar) {
		d.scalar &= scalar2u8(scalar);
		return d;
	}

	template <typename T>
	friend Derived& operator^=(Derived& d, const T& scalar) {
		d.scalar ^= scalar2u8(scalar);
		return d;
	}

	template <typename T>
	friend Derived& operator|=(Derived& d, const T& scalar) {
		d.scalar |= scalar2u8(scalar);
		return d;
	}

	template <typename T>
	friend bool operator<(const Derived& d, const T& scalar) {
		return d.scalar < scalar2u8(scalar);
	}

	template <typename T>
	friend bool operator>(const Derived& d, const T& scalar) {
		return scalar2u8(scalar) < d.scalar;
	}

	template <typename T>
	friend bool operator<=(const Derived& d, const T& scalar) {
		return !(d.scalar > scalar2u8(scalar));
	}

	template <typename T>
	friend bool operator>=(const Derived& d, const T& scalar) {
		return !(d.scalar < scalar2u8(scalar));
	}

	template <typename T>
	friend bool operator==(const Derived& d, const T& scalar) {
		return d.scalar == scalar2u8(scalar);
	}

	template <typename T>
	friend bool operator!=(const Derived& d, const T& scalar) {
		return !(d == scalar);
	}

	template <typename T>
	friend bool operator&&(const Derived& d, const T& scalar) {
		return d.scalar && scalar2u8(scalar);
	}

	template <typename T>
	friend bool operator||(const Derived& d, const T& scalar) {
		return d.scalar || scalar2u8(scalar);
	}

};

struct Red : public RGBBasicOp<Red> {
	uint8_t scalar;

	Red() : scalar(0u) {}
	template <typename T>
	Red(T scalar) : scalar(scalar2u8(scalar)) {}

	template <typename T>
	Red& operator=(const T& v)& {
		this->scalar = scalar2u8(v);
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, const Red& test) {
		out << std::to_string(test.scalar);
		return out;
	}
};

struct Green : public RGBBasicOp<Green> {
	uint8_t _dummy1{}, scalar;

	Green() : scalar(0u) {}
	template <typename T>
	Green(T scalar) : scalar(scalar2u8(scalar)) {}

	template <typename T>
	Green& operator=(const T& v)& {
		this->scalar = scalar2u8(v);
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, const Green& test) {
		out << std::to_string(test.scalar);
		return out;
	}
};

struct Blue : public RGBBasicOp<Blue> {
	uint8_t _dummy1{}, _dummy2{}, scalar;

	Blue() : scalar(0u) {}
	template <typename T>
	Blue(T scalar) : scalar(scalar2u8(scalar)) {}

	template <typename T>
	Blue& operator=(const T& v)& {
		this->scalar = scalar2u8(v);
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, const Blue& test) {
		out << std::to_string(test.scalar);
		return out;
	}
};

struct Alpha : public RGBBasicOp<Alpha> {
	uint8_t _dummy1{}, _dummy2{}, _dummy3{}, scalar;

	Alpha() : scalar(0u) {}
	template <typename T>
	Alpha(T scalar) : scalar(scalar2u8(scalar)) {}

	template <typename T>
	Alpha& operator=(const T& v)& {
		this->scalar = scalar2u8(v);
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, const Alpha& test) {
		out << std::to_string(test.scalar);
		return out;
	}
};

}

namespace kika::color {
union RGBA {
	uint32_t hex; // a-b-g-r 0xa1a2_b1b2_g1g2_r1r2
	std::array<uint8_t, 4> dec; // rgba
	std::array<uint8_t, 3> rgb; // rgb

	impl::Red r;
	impl::Green g;
	impl::Blue b;
	impl::Alpha a;

	RGBA() : dec({ 0u, 0u, 0u, 255u }) {}
	RGBA(uint32_t rgba) {
		*this = rgba;
	}

	template <typename T, class = std::enable_if_t<!std::is_scalar_v<T>, void>>
	RGBA(T rgba) {
		int i = 0;
		for (auto itr = rgba.begin(); itr != rgba.end(); ++itr) {
			dec[i++] = impl::scalar2u8(*itr);
		}
		for (; i < 3; i++) dec[i] = 0u;
		for (; i < 4; i++) dec[i] = 255u;
	}

	template <typename R, typename G, typename B, typename A=uint8_t>
	RGBA(R r, G g, B b, A a = 255u) {
		dec[0] = impl::scalar2u8(r);
		dec[1] = impl::scalar2u8(g);
		dec[2] = impl::scalar2u8(b);
		dec[3] = impl::scalar2u8(a);
	}

	RGBA& operator=(uint32_t rgba)& {
		dec[3] = (rgba >> 0u) & 0xffu;
		dec[2] = (rgba >> 8u) & 0xffu;
		dec[1] = (rgba >> 16u) & 0xffu;
		dec[0] = (rgba >> 24u) & 0xffu;
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, const RGBA& rgba) {
		out << "(" << rgba.r << "," << rgba.g << "," << rgba.b << "," << rgba.a << ")";
		return out;
	}

	friend RGBA operator*(const RGBA& color, const double t) {
		return RGBA(color.r * t, color.g * t, color.b * t, color.a * t);
	}

	friend RGBA operator+(const RGBA& color, const RGBA& other) {
		return RGBA(color.r + other.r, color.g + other.g, color.b + other.b, color.a + other.a);
	}

};

}
*/