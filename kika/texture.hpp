#pragma once

#include <kika/color.hpp>

namespace kika::tex {
struct TextureImage {
	uint32_t reso;
	std::vector<color::RGBA> img;

	struct GeometryData {
		std::array<std::array<double, 3>, 4> pos;
		std::vector<uint32_t> idx;
		std::array<std::array<double, 3>, 4> color;
		std::array<std::array<double, 3>, 4> vn;
		std::array<std::array<double, 2>, 4> uv;
	} geom;

	void prepareGeomData() {
		geom.pos[0] = std::array<double, 3>{ -0.5, -0.5, 0.0 };
		geom.pos[1] = std::array<double, 3>{ -0.5, 0.5, 0.0 };
		geom.pos[2] = std::array<double, 3>{ 0.5, 0.5, 0.0 };
		geom.pos[3] = std::array<double, 3>{ 0.5, -0.5, 0.0 };

		geom.idx = std::vector<uint32_t>{ 0, 1, 3, 1, 2, 3 };

		geom.vn[0] = std::array<double, 3>{ 0., 0., 1. };
		geom.vn[1] = std::array<double, 3>{ 0., 0., 1. };
		geom.vn[2] = std::array<double, 3>{ 0., 0., 1. };
		geom.vn[3] = std::array<double, 3>{ 0., 0., 1. };

		geom.color[0] = std::array<double, 3>{ 1, 0, 0 };
		geom.color[1] = std::array<double, 3>{ 0, 1, 0 };
		geom.color[2] = std::array<double, 3>{ 1, 1, 1 };
		geom.color[3] = std::array<double, 3>{ 0, 0, 1 };

		geom.uv[0] = std::array<double, 2>{ 0, 0 };
		geom.uv[1] = std::array<double, 2>{ 1, 0 };
		geom.uv[2] = std::array<double, 2>{ 1, 1 };
		geom.uv[3] = std::array<double, 2>{ 0, 1 };
	}

	TextureImage() : reso(0), img({}) { prepareGeomData(); }
	TextureImage(const uint32_t reso) : reso(reso), img(reso * reso, 0xfffffffful) { prepareGeomData(); }

	size_t size() const {
		return reso * reso * sizeof(uint32_t);
	}

	color::RGBA* data() {
		return img.data();
	}

	// rotate 90 (counter-clockwise)
	void rotate90c() {
		for (uint32_t i = 0; i < reso / 2; i++) {
			for (uint32_t j = i; j < reso - i - 1; j++) {
				auto tmp = img[i * reso + j];
				img[i * reso + j] = img[j * reso + reso - i - 1];
				img[j * reso + reso - i - 1] = img[(reso - i - 1) * reso + reso - j - 1];
				img[(reso - i - 1) * reso + reso - j - 1] = img[(reso - j - 1) * reso + i];
				img[(reso - j - 1) * reso + i] = tmp;
			}
		}
	}

	void upsideDown() {
		for (uint32_t i = 0; i < reso; i++) {
			for (uint32_t j = 0; j < reso / 2; j++) {
				std::swap(img[i * reso + j], img[i * reso + reso - j - 1]);
			}
		}
	}

	// luminosity base method
	void grayscale() {
		for (uint32_t i = 0; i < reso * reso; i++) {
			img[i] = img[i].grayscale();
		}
	}

	void gammaCorrection() {
		for (uint32_t i = 0; i < reso * reso; i++) {
			img[i] = img[i].gammaCorrection();
		}
	}

	TextureImage mix(const TextureImage& other, double t) {
		t = (t < 0. ? 0. : (t > 1. ? 1. : t));
		TextureImage ret(reso);
		for (size_t i = 0; i < (reso * reso); i++) {
			ret.img[i] = img[i].mix(other.img[i], t);
		}
		return ret;
	}

	TextureImage& textureImage() {
		return *this;
	}

	color::RGBA sample(double u, double v) {
		u = fmod(fmod(u, 1.0) + 1.0, 1.0);
		v = fmod(fmod(v, 1.0) + 1.0, 1.0);

		double step = 1. / (reso - 1);
		double dx = fmod(u, step) / step, dy = fmod(v, step) / step;
		uint32_t lx = static_cast<uint32_t>(floor(u * (reso-1))), 
			ly = static_cast<uint32_t>(floor(v * (reso-1))),
			rx = lx + (lx != reso), ry = ly + (ly != reso);

		return img[lx * reso + ly].mix(img[rx * reso + ly], dx).mix(
		img[lx * reso + ry].mix(img[rx * reso + ry], dx), dy);
	}

};

class Checker : public TextureImage {
public:
	std::vector<color::RGBA> colors;

	Checker() : colors({ 0xf3efe0ffu, 0x561066ffu }) {}
	template <typename ...RGBAs>
	Checker(color::RGBA color1, color::RGBA color2, RGBAs... colors) : colors({ color1, color2, colors... }) {}

	// Checker(ramp, step).create(reso, step)
	// tile image : Checker(ramp, step(odd)).create(reso, (step+1) / 2)
	Checker(const color::ColorRamp& ramp, const size_t step) {
		for (size_t i = 0; i < step; i++) {
			colors.emplace_back(ramp((double)i / (step - 1)));
		}
	}

	Checker& create(const uint32_t reso = 512, const uint32_t repeat = 32) {
		auto itr = colors.begin(), bitr = itr;
		this->reso = reso;
		img.resize(reso * reso, *itr);

		double k = (double)reso / repeat;
		double si = 0, sj = 0;
		for (uint32_t i = 0; i < reso; i++) {
			itr = bitr;
			sj = 0;
			for (uint32_t j = 0; j < reso; j++) {
				img[i * reso + j] = *itr;
				sj += 1;
				if (sj > k) {
					sj = fmod(sj, k);
					++itr;
					if (itr == colors.end()) itr = colors.begin();
				}
			}
			si += 1;
			if (si > k) {
				si = fmod(si, k);
				++bitr;
				if (bitr == colors.end()) bitr = colors.begin();
			}
		}
		return *this;
	}
};

class Gradation : public TextureImage {
public:
	color::ColorRamp ramp;
	
	Gradation() : ramp{ color::palette::coolWarm() } {}
	template <typename ...RGBAs>
	Gradation(color::RGBA color1, color::RGBA color2, RGBAs... colors) : ramp{ { color1, color2, colors... } } {}
	Gradation(const color::ColorRamp& colorRamp) : ramp(colorRamp) {}

	Gradation& create(const uint32_t reso = 512) {
		color::RGBA alpha = ramp(0.0);
		this->reso = reso;
		img.resize(reso * reso, alpha);

		for (uint32_t i = 0; i < reso; i++) {
			auto c = ramp((double) i / (reso - 1));
			for (uint32_t j = 0; j < reso; j++) {
				img[j * reso + i] = c;
			}
		}
		return *this;
	}
};

class Noise : public TextureImage {
public:
	Noise() {}

	Noise& create(const uint32_t reso = 512, const uint32_t level = 0, const unsigned int seed = std::random_device{}()) {
		this->reso = reso;
		img.resize(reso * reso, 0x0);

		std::mt19937 engine(seed);
		std::uniform_int_distribution<uint32_t> distrib(0u, 0xffffffffu);

		uint32_t stride = 1u << level;
		for (uint32_t i = 0; i < reso; i+=stride) {
			for (uint32_t j = 0; j < reso; j+=stride) {
				uint32_t c = distrib(engine);
				for (uint32_t k = 0; k < stride; k++) {
					for (uint32_t r = 0; r < stride; r++) {
						img[(i + k) * reso + j + r].r() = (uint8_t)((c & 0x000000ffu) >> 0u);
						img[(i + k) * reso + j + r].g() = (uint8_t)((c & 0x0000ff00u) >> 8u);
						img[(i + k) * reso + j + r].b() = (uint8_t)((c & 0x00ff0000u) >> 16u);
					}
				}
			}
		}

		return *this;
	}
};

}
