#pragma once
#include <numeric>
#include <vector>
#include <string>

namespace kika::shape {

constexpr double _PI = 3.14159265359;

enum StorageOptions : int {
	ColMajor = 0,
	RowMajor = 1,
};

using enum StorageOptions;

template <typename T, typename Idx>
struct ShapeIterator {
	using difference_type = std::ptrdiff_t;
	using value_type = T;
	using reference = T&;
	using pointer = T*;
	using iterator_category = std::forward_iterator_tag;

	std::vector<T> data;
	Idx i;

	ShapeIterator(const std::vector<T>& data) : data(data), i(0) {}
	ShapeIterator(const Idx& i) : data({}), i(i) {}

	reference operator*() noexcept {
		return data[i];
	}

	ShapeIterator& operator++() {
		++i;
		return *this;
	}

	ShapeIterator& operator++(int) {
		auto temp = *this;
		++* this;
		return temp;
	}

	bool operator==(const ShapeIterator& itr) const {
		return i == itr.i;
	}

	bool operator!=(const ShapeIterator& itr) const {
		return i != itr.i;
	}

};

// Basis class.
template <typename Prec=double, typename Idx=uint32_t>
class Shape {
protected:
	template <typename Params, typename T>
	struct Generator {
	protected:
		const Params& params;

		Generator(const Params& params) : params(params) {}
		
	public:
		virtual Idx size() const = 0;

		virtual std::vector<T> map(const int storage = StorageOptions::RowMajor) const = 0;

		ShapeIterator<T, Idx> begin(const int storage = StorageOptions::RowMajor) const {
			return ShapeIterator<T, Idx>(map(storage));
		}

		ShapeIterator<T, Idx> end() const {
			return ShapeIterator<T, Idx>(size());
		}
	};

public:
	using value_type = Prec;
	using index_type = Idx;

	Shape() {}
	virtual ~Shape() {}

	virtual std::vector<std::vector<Idx>> poly() const = 0;

	// number of vertices
	virtual Idx nv() const = 0;

	// number of faces (triangle mesh)
	virtual Idx nf() const = 0;

	// number of edges (triangle mesh)
	virtual Idx ne() const = 0;

};

template <typename Prec=double, typename Idx=uint32_t>
class PresetShape : public Shape<Prec, Idx> {

protected:
	std::vector<Prec> xyzBuffer;
	std::vector<Idx> idxBuffer;

	using XyzGeneratorBase = typename Shape<Prec, Idx>::template Generator<bool, Prec>;
	using TriGeneratorBase = typename Shape<Prec, Idx>::template Generator<bool, Idx>;

	class XyzGenerator : public XyzGeneratorBase {
		using XyzGeneratorBase::params;
		std::reference_wrapper<const std::vector<Prec>> xyzBuffer;
		bool _dummy;

	public:
		XyzGenerator(const std::vector<Prec>& xyzBuffer) : XyzGeneratorBase(_dummy), xyzBuffer(xyzBuffer) {}

		Idx size() const override {
			return static_cast<Idx>(xyzBuffer.get().size());
		};

		std::vector<Prec> map(const int storage = StorageOptions::RowMajor) const override {
			if (storage == StorageOptions::RowMajor) {
				return xyzBuffer;
			} else {
				Idx n = size() / 3;
				std::vector<Prec> pos(3 * n);
				for (Idx i = 0; i < n; i++) {
					pos[i + 0] = xyzBuffer.get()[i * 3 + 0];
					pos[i + n] = xyzBuffer.get()[i * 3 + 1];
					pos[i + 2*n] = xyzBuffer.get()[i * 3 + 2];
				}
				return pos;
			}
		}
	};

	class TriGenerator : public TriGeneratorBase {
		using TriGeneratorBase::params;

		std::reference_wrapper<const std::vector<Idx>> idxBuffer;
		bool _dummy;

	public:
		TriGenerator(const std::vector<Idx>& idxBuffer) : TriGeneratorBase(_dummy), idxBuffer(idxBuffer) {}

		Idx size() const override {
			return static_cast<Idx>(idxBuffer.get().size());
		};

		std::vector<Idx> map(const int storage = StorageOptions::RowMajor) const override {
			if (storage == StorageOptions::RowMajor) {
				return idxBuffer;
			} else {
				Idx n = size() / 3;
				std::vector<Idx> idx(3 * n);
				for (Idx i = 0; i < n; i++) {
					idx[i + 0] = idxBuffer.get()[i * 3 + 0];
					idx[i + n] = idxBuffer.get()[i * 3 + 1];
					idx[i + 2*n] = idxBuffer.get()[i * 3 + 2];
				}
				return idx;
			}
		}
	};

public:
	XyzGenerator xyz;
	TriGenerator tri;

protected:

	PresetShape(std::string_view filename) : xyzBuffer({}), idxBuffer({}), xyz(xyzBuffer), tri(idxBuffer) {
		loadXyz(filename.data() + std::string("_raw_xyz"), StorageOptions::RowMajor);
		loadIdx(filename.data() + std::string("_raw_idx"), StorageOptions::RowMajor);
	}

	void loadXyz(std::string_view filename, const int storage = StorageOptions::RowMajor) {
		std::ifstream file(filename.data(), std::ios::in | std::ios::binary);

		if (file.fail()) {
			std::cerr << "Failed to open raw_data file.\n" << "Please check /raw_data/path.\n";
			return;
		}

		uint32_t n;
		file.read((char*)(&n), sizeof(uint32_t));

		std::vector<float> data;
		data.resize(3 * n);
		file.read((char*)data.data(), 3*n*sizeof(float));
		file.close();

		uint32_t k1 = (storage == StorageOptions::ColMajor) ? 1 : 3;
		uint32_t k2 = (storage == StorageOptions::ColMajor) ? n : 1;

		xyzBuffer.resize(3 * n);

		for (uint32_t i = 0; i < n; i++) {
			xyzBuffer[i * k1 + 0] = static_cast<Prec>(data[i + 0]);
			xyzBuffer[i * k1 + k2] = static_cast<Prec>(data[i + n]);
			xyzBuffer[i * k1 + 2*k2] = static_cast<Prec>(data[i + 2*n]);
		}

	}

	void loadIdx(std::string_view filename, const int storage = StorageOptions::RowMajor) {
		std::ifstream file(filename.data(), std::ios::in | std::ios::binary);

		if (file.fail()) {
			std::cerr << "Failed to open raw_data file.\n" << "Please check /raw_data/path.\n";
			return;
		}

		uint32_t n, f;
		file.read((char*)(&n), sizeof(uint32_t));
		file.read((char*)(&f), sizeof(uint32_t));

		std::vector<uint32_t> data;
		data.resize(n);
		file.read((char*)data.data(), n*sizeof(uint32_t));
		file.close();

		int stripType = 0;

		idxBuffer.resize(f);
		uint32_t k1 = (storage == StorageOptions::ColMajor) ? 1 : 3;
		uint32_t k2 = (storage == StorageOptions::ColMajor) ? f / 3 : 1;

		uint32_t c = 0;
		for (size_t i = 0; i < data.size();) {
			if (data[i] == 0 || data[i] == 1) {
				stripType = data[i];
				i++;
			} else {
				if (stripType) {
					if (data[i+2] != 0 && data[i+2] != 1 && data[i+1] != 0 && data[i+1] != 1) {
						idxBuffer[c * k1 + 0] = static_cast<Idx>(data[i+0] - 2);
					if (stripType == 2) {
							idxBuffer[c * k1 + k2] = static_cast<Idx>(data[i+2] - 2);
							idxBuffer[c++ * k1 +2*k2] = static_cast<Idx>(data[i+1] - 2);
						} else {
							idxBuffer[c * k1 + k2] = static_cast<Idx>(data[i+1] - 2);
							idxBuffer[c++ * k1 + 2*k2] = static_cast<Idx>(data[i+2] - 2);
						}
						stripType = stripType % 2 + 1;
					}
					i++;
				} else {
					idxBuffer[c * k1 + 0] = static_cast<Idx>(data[i+0] - 2);
					idxBuffer[c * k1 + k2] = static_cast<Idx>(data[i+1] - 2);
					idxBuffer[c++ * k1 + 2*k2] = static_cast<Idx>(data[i+2] - 2);
					i += 3;
				}
			}
		}

	}

public:

	Idx nv() const override final {
		return static_cast<Idx>(xyzBuffer.size() / 3);
	}

	Idx nf() const override final {
		return static_cast<Idx>(idxBuffer.size() / 3);
	}

	std::vector<std::vector<Idx>> poly() const {
		std::vector<std::vector<Idx>> idx(nf(), std::vector<Idx>(3));

		for (size_t i = 0; i < idx.size(); i++) {
			idx[i][0] = idxBuffer[i*3 + 0];
			idx[i][1] = idxBuffer[i*3 + 1];
			idx[i][2] = idxBuffer[i*3 + 2];
		}

		return idx;
	}

};

}
