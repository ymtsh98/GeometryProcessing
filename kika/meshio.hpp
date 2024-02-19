#pragma once
#include <iostream>
#include <string>
#include <tuple>
#include <array>
#include <vector>
#include <kika/kikautil.hpp>
#include <kika/lp_file.hpp>

namespace kika::util {
template <typename Idx=uint32_t, typename T = std::vector<uint32_t>>
std::vector<Idx> faces2indices(const std::vector<T>& faces) {
	std::vector<Idx> indices;
	indices.reserve(faces.size() * 3);
	for (auto& f : faces) {
		int i = f[0], t = f[1];
		for (size_t j = 2; j < f.size(); j++) {
			indices.emplace_back(static_cast<Idx>(i));
			indices.emplace_back(static_cast<Idx>(t));
			indices.emplace_back(static_cast<Idx>(f[j]));
			t = f[j];
		}
	}
	return indices;
}

template <typename Idx=uint32_t, typename T=uint32_t>
std::vector<std::vector<Idx>> indices2faces(const std::vector<T>& indices) {
	std::vector<std::vector<Idx>> faces;
	faces.resize(indices.size() / 3, std::vector<Idx>(3));
	for (size_t i = 0; i < faces.size(); i++) {
		faces[i][0] = static_cast<Idx>(indices[i*3 + 0]);
		faces[i][1] = static_cast<Idx>(indices[i*3 + 1]);
		faces[i][2] = static_cast<Idx>(indices[i*3 + 2]);
	}
	return faces;
}
}

namespace kika::mio {
template<typename T1 = std::array<double, 3>, typename T2 = std::vector<uint32_t>>
std::tuple<std::vector<T1>, std::vector<T2>> loadOFFvf(const std::string& path) {
	std::tuple<std::vector<T1>, std::vector<T2>> ret;
	std::vector<T1>& vertices = std::get<0>(ret);
	std::vector<T2>& indices = std::get<1>(ret);

	size_t numVerts = -1, numFaces = -1;

	kika::util::LPFile f(path);

	while (f.getline() != "OFF") {}

	vertices.resize(std::stoi(f.getseg()), T1{ 0, 0, 0 });
	indices.resize(std::stoi(f.getseg()));
	f.skipline();
	for (size_t i = 0; i < vertices.size(); i++) {
		vertices[i][0] = static_cast<typename T1::value_type>(std::stod(f.getseg()));
		vertices[i][1] = static_cast<typename T1::value_type>(std::stod(f.getseg()));
		vertices[i][2] = static_cast<typename T1::value_type>(std::stod(f.getseg()));
	}

	for (size_t i = 0; i < indices.size(); i++) {
		auto s = std::stoi(f.getseg());
		indices[i] = T2(s);
		for (int j = 0; j < s; j++)
			indices[i][j] = static_cast<typename T2::value_type>(std::stoi(f.getseg()));
	}

	return ret;
}


template <typename T1 = double, typename T2 = uint32_t>
class OBJReader {

	template <size_t Dim>
	class VFData {
	public:
		bool isRowMajor;
		std::vector<T1> verts;

		size_t nv;
		std::vector<std::vector<T2>> faces;

		VFData() : isRowMajor(true), nv(0), verts({}), faces({}) {}

		template <bool RowMajor = false>
		constexpr T1* vdata() {
			if constexpr (RowMajor) {
				if (isRowMajor) return verts.data();

				util::rowMajorOrder<Dim>(nv, verts);
				isRowMajor = !isRowMajor;
				return verts.data();
			} else {
				if (!isRowMajor) return verts.data();

				util::colMajorOrder<Dim>(nv, verts);
				isRowMajor = !isRowMajor;
				return verts.data();
			}
		}
	};

public:

	VFData<3> pos;
	VFData<2> uv;
	std::vector<T2> uv2pos;

	OBJReader() : pos({}), uv({}), uv2pos({}) {}

	OBJReader(std::string_view path) : OBJReader() {
		read(path);
	}

	template <bool UV = false>
	OBJReader& read(std::string_view path) {
		kika::util::LPFile f(path);

		size_t pred = f.size / 32 / 3;
		pos.verts.reserve(pred * 3);
		pos.faces.reserve(pred * 2);

		std::vector<T2> pfi, tfi;
		pfi.reserve(4);

		if constexpr (UV) {
			uv.verts.reserve(pred * 2);
			uv.faces.reserve(pred * 2);
			tfi.reserve(4);
		}

		while (!f.eof()) {
			auto s = f.curtseg();

			if (s == "v") {
				f.skipseg();
				pos.verts.emplace_back(static_cast<T1>(stod(f.getseg())));
				pos.verts.emplace_back(static_cast<T1>(stod(f.getseg())));
				pos.verts.emplace_back(static_cast<T1>(stod(f.getseg())));
				pos.nv++;
			} else if (s == "f") {
				f.skipseg();

				pfi.clear();
				if constexpr (UV) tfi.clear();
				bool endl = false;

				while (!endl) {
					size_t left, right;

					auto seg = f.getseg(endl);
					left = seg.find('/');
					int pi = (left == std::string::npos) ? std::stoi(seg) : std::stoi(seg.substr(0, left));
					pi = (pi < 0 ? pi + (int)pos.nv : pi) - 1;
					pfi.emplace_back(static_cast<T2>(pi));

					if constexpr (UV) {
						if (left == std::string::npos) continue;
						right = seg.rfind('/');
						int ti = std::stoi(seg.substr(left+1, right));
						ti = ti < 0 ? ti + (int)uv.nv : ti - 1;
						tfi.emplace_back(static_cast<T2>(ti));
						uv2pos[ti] = static_cast<T2>(pi);
					}
				}
				pos.faces.emplace_back(pfi);
				if constexpr (UV) uv.faces.emplace_back(tfi);
			} else if constexpr (UV) {
				if (s == "vt") {
					f.skipseg();
					uv.verts.emplace_back(static_cast<T1>(stod(f.getseg())));
					uv.verts.emplace_back(static_cast<T1>(stod(f.getseg())));
					uv2pos.emplace_back(static_cast<T2>(uv2pos.size()));
					uv.nv++;
				} else f.skipline();
			} else {
				f.skipline();
			}
		}

		return *this;
	}

};

template<typename T1 = std::array<double, 3>, typename T2 = uint32_t, typename F= typename T1::value_type>
std::tuple<std::vector<T1>, std::vector<std::vector<T2>>> loadOBJvf(const std::string& path) {
	std::tuple<std::vector<T1>, std::vector<std::vector<T2>>> ret;
	std::vector<T1>& vertices = std::get<0>(ret);
	std::vector<std::vector<T2>>& faces = std::get<1>(ret);

	kika::util::LPFile f(path);
	vertices.reserve(f.size / 32 / 3);
	faces.reserve(f.size / 32 / 3 * 2);

	std::vector<T2> fi;
	fi.reserve(4);

	while (!f.eof()) {
		auto s = f.curtseg();

		if (s == "v") {
			f.skipseg();
			vertices.emplace_back(T1{ static_cast<F>(stod(f.getseg())),
				static_cast<F>(stod(f.getseg())), static_cast<F>(stod(f.getseg())) });
		} else if (s == "f") {
			f.skipseg();
		
			fi.clear();
			bool endl = false;
			while (!endl) {
				auto seg = f.getseg(endl);
				auto left = seg.find('/');
				int pi;
				if (left == std::string::npos) pi = std::stoi(seg);
				else pi = std::stoi(seg.substr(0, left));
				pi = pi < 0 ? pi + (int)vertices.size() : pi - 1;
				fi.emplace_back(static_cast<T2>(pi));
			}
			faces.emplace_back(fi);
		} else {
			f.skipline();
		}
	}

	return ret;
}

class OBJWriter {
	std::string wr;
	size_t offset;

public:
	OBJWriter() : wr(), offset(1) {}

	template<typename T1 = std::vector<double>, typename T2 = std::vector<uint32_t>>
	void write(const T1& vertices ={}, const T2& faces ={}) {
		auto n = vwrite(vertices);
		fwrite(faces, n);
	}

	template <typename T = std::vector<double>>
	size_t vwrite(const T& vertices) {
		size_t n = 0;

		if constexpr (util::IsEigen<T>KIKA_VALUE) {
			n = (size_t)vertices.rows();
			wr.reserve(wr.size() + n * 30);
			for (size_t i = 0; i < n; i++) {
				wr += std::string("v " + std::to_string(vertices(i, 0)) + " " + std::to_string(vertices(i, 1)) + " " + std::to_string(vertices(i, 2)) + "\n");
			}
		} else if constexpr (util::IsStdvec2d<T>KIKA_VALUE) {
			n = vertices.size();
			wr.reserve(wr.size() + n * 30);
			for (size_t i = 0; i < n; i++) {
				wr += std::string("v " + std::to_string(vertices[i][0]) + " " + std::to_string(vertices[i][1]) + " " + std::to_string(vertices[i][2]) + "\n");
			}
		} else {
			n = vertices.size() / 3;
			wr.reserve(wr.size() + n * 30);
			for (size_t i = 0; i < vertices.size(); i+=3) {
				wr += std::string("v " + std::to_string(vertices[i+0]) + " " + std::to_string(vertices[i+1]) + " " + std::to_string(vertices[i+2]) + "\n");
			}
		}

		return n;
	}

	template<typename T = std::vector<uint32_t>>
	void fwrite(const T& faces, const size_t addOffset = 0) {
		if constexpr (util::IsEigen<T> KIKA_VALUE) {
			wr.reserve(wr.size() + faces.rows() * 30);
			for (int i = 0; i < faces.rows(); i++) {
				wr += std::string("f " + std::to_string(faces(i, 0)+offset) + " " + std::to_string(faces(i, 1)+offset) + " " + std::to_string(faces(i, 2)+offset) + "\n");
			}
		} else if constexpr (util::IsStdvec2d<T> KIKA_VALUE) {
			wr.reserve(wr.size() + faces.size() * 30);
			for (size_t i = 0; i < faces.size(); i++) {
				wr += "f ";
				for (size_t j = 0; j < faces[i].size(); j++)
					wr += std::to_string(faces[i][j]+offset) + (j + 1 != faces[i].size() ? " " : "\n");
			}
		} else {
			wr.reserve(wr.size() + faces.size() * 10);
			for (size_t i = 0; i < faces.size(); i+=3) {
				wr += std::string("f " + std::to_string(faces[i+0]+offset) + " " + std::to_string(faces[i+1]+offset) + " " + std::to_string(faces[i+2]+offset) + "\n");
			}
		}

		offset += addOffset;
	}

	template <typename T = std::vector<uint32_t>>
	void lwrite(const T& line, const size_t addOffset = 0) {
		for (size_t i = 0, j = line.size(); i < j; i++) {
		  wr += ("l " + std::to_string(line[i]+offset) + " " + std::to_string(line[(i + 1) % j]+offset) + "\n");
		}

		offset += addOffset;
	}

	template <typename T = std::vector<uint32_t>>
	void bspwrite(const T& line, const size_t addOffset = 0) {
		wr += "cstype bspline\n";
		wr += "deg 2\n";
		wr += "curv 0.00 1.00 ";
		for (size_t i = 0; i < line.size(); i++) wr += std::to_string(line[i] + 1) + " ";
		wr += std::to_string(line[0] + 1) + " " + std::to_string(line[1] + 1) + "\n";
		wr += "parm u ";
		for (size_t i = 0; i < line.size() + 2; i++)
			wr += std::to_string((double)i / (line.size() + 2)) + " ";
		wr += "1.00\nend\n";
		wr += "\n";

		offset += addOffset;
	}


	void swrite(const std::string& str) {
		wr += str;
	}

	void save(const std::string_view fname, const bool clear = true) {
		std::ofstream out(fname.data());
		out.write(wr.data(), wr.size());
		out.close();

		if (clear) {
			wr.clear();
			wr.shrink_to_fit();
		}
	}
};

template<typename T1, typename T2>
void saveOBJ(std::string_view saveName, const T1& vertices, const T2& faces) {
	OBJWriter writer;
	writer.write(vertices, faces);
	writer.save(saveName);
}

}