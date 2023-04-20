#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <mygp/lp_file.hpp>

namespace gp::util {
inline std::vector<std::string> split(const std::string& str, const char delim = ' ') {
	std::istringstream is(str);
	std::vector<std::string> results;
	std::string s;

	while (std::getline(is, s, delim)) {
		results.emplace_back(s);
	}
	return results;
}

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

inline bool checkFileExtension(std::string_view path, std::string_view fext) {
	return path.length() > fext.length() &&
		std::equal(path.rbegin(), path.rbegin() + fext.length(), fext.rbegin());
}
}

namespace gp::mio::impl {
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
}

namespace gp::mio {
template<typename T1 = std::vector<double>, typename T2 = std::vector<int>>
std::tuple<std::vector<T1>, std::vector<T2>> loadOFFvf(const std::string& path) {
	std::tuple<std::vector<T1>, std::vector<T2>> ret;
	std::vector<T1>& vertices = std::get<0>(ret);
	std::vector<T2>& indices = std::get<1>(ret);

	size_t numVerts = -1, numFaces = -1;

	gp::util::LPFile f(path);

	while (f.getline() != "OFF") {}

	vertices.resize(std::stoi(f.getseg()), T1{ 0, 0, 0 });
	indices.resize(std::stoi(f.getseg()));
	f.skipline();
	for (size_t i = 0; i < vertices.size(); i++) {
		for (size_t j = 0; j < 3; j++) 
			vertices[i][j] = static_cast<T1::value_type>(std::stod(f.getseg()));
	}

	for (size_t i = 0; i < indices.size(); i++) {
		auto s = std::stoi(f.getseg());
		indices[i] = T2(s);
		for (int j = 0; j < s; j++)
			indices[i][j] = std::stoi(f.getseg());
	}

	return ret;
}

template<typename T1 = std::vector<double>, typename T2 = int, typename F= typename T1::value_type>
std::tuple<std::vector<T1>, std::vector<std::vector<T2>>> loadOBJvf(const std::string& path) {
	std::tuple<std::vector<T1>, std::vector<std::vector<T2>>> ret;
	std::vector<T1>& vertices = std::get<0>(ret);
	std::vector<std::vector<T2>>& faces = std::get<1>(ret);

	gp::util::LPFile f(path);
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
				fi.emplace_back(pi);
			}
			faces.emplace_back(fi);
		} else {
			f.skipline();
		}
	}

	return ret;

	//// old version
	/*
	std::ifstream f(path, std::ios::ate);
	if (f.fail()) { std::cerr << "Failed to open file" << std::endl;  exit(1); };

	size_t fsize = (size_t)f.tellg();
	f.seekg(0);
	std::vector<char> text(fsize);

	f.read(text.data(), fsize);
	f.close();

	size_t seek = 0;
	std::string line = "";
	line.reserve(200);
	std::vector<std::string> sline;
	sline.reserve(4);

	std::vector<T2> fi; fi.reserve(3);

	while (seek < fsize) {
		line.clear();
		sline.clear();
		while (seek < fsize && text[seek] != '\n') {
			if (text[seek] == ' ' || text[seek] == '	') {
				if (line != "") sline.emplace_back(line);
				line = "";
			} else {
				line += text[seek];
			}
			seek++;
		}

		if (line != "") sline.emplace_back(line);
		seek++;

		if (sline.size() < 2 || sline[0][0] == '#') continue;

		if (sline[0] == "v") {
			vertices.emplace_back(T1{ static_cast<F>(stod(sline[1])),
				static_cast<F>(stod(sline[2])), static_cast<F>(stod(sline[3])) });
		} else if (sline[0] == "f") {
			fi.clear();

			for (size_t i = 1; i < sline.size(); i++) {
				auto left = sline[i].find('/');
				int pi;

				if (left == std::string::npos) pi = atoi(sline[i].data());
				else 	pi = atoi(sline[i].substr(0, left).data());
				pi = pi < 0 ? pi + (int)vertices.size() : pi - 1;

				fi.emplace_back(pi);
			}
			faces.emplace_back(fi);
		}
	}

	return ret;
	*/
}

template<typename T1, typename T2>
void saveOBJ(std::string_view saveName, const T1& vertices, const T2& faces) {
	std::ofstream f(saveName.data());
	std::string wr = "";

	if constexpr (impl::IsEigen<T1>::value) {
		wr.reserve(vertices.rows());
		for (int i = 0; i < vertices.rows(); i++) {
			wr += std::string("v " + std::to_string(vertices(i, 0)) + " " + std::to_string(vertices(i, 1)) + " " + std::to_string(vertices(i, 2)) + "\n");
		}
	} else if constexpr (impl::IsStdvec2d<T1>::value) {
		wr.reserve(vertices.size());
		for (size_t i = 0; i < vertices.size(); i++) {
			wr += std::string("v " + std::to_string(vertices[i][0]) + " " + std::to_string(vertices[i][1]) + " " + std::to_string(vertices[i][2]) + "\n");
		}
	} else {
		wr.reserve(vertices.size() * 3);
		for (size_t i = 0; i < vertices.size(); i+=3) {
			wr += std::string("v " + std::to_string(vertices[i+0]) + " " + std::to_string(vertices[i+1]) + " " + std::to_string(vertices[i+2]) + "\n");
		}
	}

	if constexpr (impl::IsEigen<T2>::value) {
		wr.reserve(faces.rows());
		for (int i = 0; i < faces.rows(); i++) {
			wr += std::string("f " + std::to_string(faces(i, 0)+1) + " " + std::to_string(faces(i, 1)+1) + " " + std::to_string(faces(i, 2)+1) + "\n");
		}
	} else if constexpr (impl::IsStdvec2d<T2>::value) {
		wr.reserve(faces.size());
		for (size_t i = 0; i < faces.size(); i++) {
			wr += "f ";
			for (size_t j = 0; j < faces[i].size(); j++)
				wr += std::to_string(faces[i][j]+1) + (j + 1 != faces[i].size() ? " " : "\n");
		}
	} else {
		wr.reserve(faces.size() * 3);
		for (size_t i = 0; i < faces.size(); i+=3) {
			wr += std::string("f " + std::to_string(faces[i+0]+1) + " " + std::to_string(faces[i+1]+1) + " " + std::to_string(faces[i+2]+1) + "\n");
		}
	}
	
	f << wr;
	f.close();
}

}