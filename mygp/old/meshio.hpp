#pragma once
#include <omp.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

namespace mio {
	template <typename T>
	std::vector<uint32_t> faces2indices(const T& faces) {
		std::vector<uint32_t> indices;
		indices.reserve(faces.size() * 3);
		for (auto& f : faces) {
			if (f.size() == 3) {
				for (auto& i : f) indices.push_back(i);
			} else if (f.size() > 3) {
				auto i = f[0], t = f[1];
				for (auto r : f) {
					indices.push_back(i);
					indices.push_back(t);
					indices.push_back(r);
					t = r;
				}
			}
		}
		return indices;
	}

	template <typename T>
	std::vector<std::vector<uint32_t>> indices2faces(const T& indices) {
		std::vector<std::vector<uint32_t>> faces;
		faces.resize(indices.size() / 3, std::vector<uint32_t>(3));
		for (size_t i = 0, t = 0; i < indices.size(); i += 3, t++) {
			faces[t] = { indices[i], indices[i + 1], indices[i + 2] };
		}
		return faces;
	}

	std::vector<std::string> split(const std::string& str, const char delim = ' ') {
		std::istringstream is(str);
		std::vector<std::string> results;
		std::string s;

		while (std::getline(is, s, delim)) {
			results.push_back(s);
		}
		return results;
	}

	template<typename T1 = std::vector<double>, typename T2 = std::vector<int>>
	std::tuple<std::vector<T1>, std::vector<T2>> loadOFF(const std::string & path) {
		std::vector<T1> vertices;
		std::vector<T2> indices;

		size_t num_verts = -1, num_faces = -1;

		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cout << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			size_t i = (seek != 0);
			std::string line = "";

			while (seek + i < fsize && text[seek + i] != '\n') {
				line += text[seek + i++];
			}

			seek += i;

			auto sline = split(line);
			if (sline.size() < 3 || sline[0] == "#") continue;

			if (num_verts == -1 && num_faces == -1) {
				num_verts = stoi(sline[0]), num_faces = stoi(sline[1]);
				vertices.reserve(num_verts);
				indices.reserve(num_faces);
			} else {
				if (sline.size() == 3) {
					vertices.push_back({ stod(sline[0]), stod(sline[1]), stod(sline[2]) });
				} else {
					indices.push_back(T2(stoi(sline[0])));
					for (size_t t = 0; t < stoi(sline[0]); t++)
						indices.back()[t] = stoi(sline[t + 1]);
				}
			}
		}
		return { vertices, indices };
	}

	template<typename T>
	void loadVert(const std::string& path, T& vertices) {
		std::ifstream f(path, std::ios::ate);
		if (f.fail()) std::cout << "failed to open file : " << path << std::endl;

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		uint32_t seek = 0;
		while (seek < fsize) {
			uint32_t i = (seek != 0);
			std::string line = "";
			std::vector<double> v; v.reserve(3);

			while (seek + i < fsize && text[seek + i] != '\n') {
				if (text[seek + i] == ' ') {
					v.push_back(stod(line));
					line = "";
				} else {
					line += text[seek + i];
				}
				i++;
			}

			seek += i;
			try {
				v.push_back(stod(line));
			} catch (std::invalid_argument& e) {
				continue;
			}
			vertices.push_back({ v[0], v[1], v[2] });
		}
	}

	template<typename T>
	void loadTri(const std::string& path, T& indices, const int offset = 0) {
		std::ifstream f(path, std::ios::ate);
		if (f.fail()) std::cout << "failed to open file : " << path << std::endl;
		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		uint32_t seek = 0;
		while (seek < fsize) {
			uint32_t i = (seek != 0);
			std::string line = "";
			std::vector<int> v; v.reserve(3);

			while (seek + i < fsize && text[seek + i] != '\n') {
				if (text[seek + i] == ' ') {
					v.push_back(stoi(line) - offset);
					line = "";
				}
				line += text[seek + i];
				i++;
			}
			seek += i;
			try {
				v.push_back(stoi(line) - offset);
			} catch (std::invalid_argument& e) {
				continue;
			}
			indices.push_back({ v[0], v[1], v[2] });
		}
	}

	template<typename T1, typename T2>
	void loadOBJ(const std::string& path, std::vector<T1>& vertices, std::vector<T2>& indices) {
		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cout << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			size_t i = (seek != 0);
			std::string line = "";

			while (seek + i < fsize && text[seek + i] != '\n') {
				line += text[seek + i++];
			}

			seek += i;

			auto sline = split(line);
			if (sline.size() < 2 || sline[0] == "#") continue;

			if (sline[0] == "v") {
				vertices.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "f") {
				indices.push_back(T2(3));
				for (size_t i = 1; i < 4; i++) {
					indices.back()[i - 1] = stoi(split(sline[i], '/')[0]) - 1;
				}
			}
		}
	}	
	
	template<typename T1, typename T2, typename T3>
	void loadOBJ(const std::string& path, std::vector<T1>& vertices, std::vector<T2>& indices, std::vector<T3>& uv) {
		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cout << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			size_t i = (seek != 0);
			std::string line = "";

			while (seek + i < fsize && text[seek + i] != '\n') {
				line += text[seek + i++];
			}

			seek += i;

			auto sline = split(line);
			if (sline.size() < 2 || sline[0] == "#") continue;

			if (sline[0] == "v") {
				vertices.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "f") {
				indices.push_back(T2(3));
				for (size_t i = 1; i < 4; i++) {
					indices.back()[i - 1] = stoi(split(sline[i], '/')[0]) - 1;
				}
			} else if (sline[0] == "vt") {
				uv.push_back({ stod(sline[1]), stod(sline[2]) });
			}
		}
	}

	template<typename T1 = std::vector<double>, typename T2 = int>
	std::tuple<std::vector<T1>, std::vector<std::vector<T2>>> loadOBJvf(const std::string path) {
		std::vector<T1> vertices;
		std::vector<std::vector<T2>> faces;

		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cerr << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			std::string line = "";
			std::vector<std::string> sline; sline.reserve(4);
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
				vertices.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "f") {
				std::vector<T2> f; f.reserve(3);

				for (size_t i = 1; i < sline.size(); i++) {
					auto left = sline[i].find('/');
					int pi;

					if (left == std::string::npos) pi = stoi(sline[i]);
					else 	pi = stoi(sline[i].substr(0, left));
					pi = pi < 0 ? pi + vertices.size() : pi - 1;

					f.push_back(pi);
				}
				faces.emplace_back(f);
			}
		}

		return { vertices, faces };
	}

	template<typename T1 = std::vector<double>, typename T2 = int>
	std::tuple<std::vector<T1>,
		std::vector<std::vector<T2>>,
		std::vector<T1>> loadOBJvfn(const std::string path) {

		std::vector<T1> vertices;
		std::vector<std::vector<T2>> faces;
		std::vector<T1> v_normals;

		std::vector<T1> positions;
		std::vector<T1> normals;
		std::map<std::pair<int, int>, size_t> unique_verts;
		size_t index = 0;

		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cerr << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);
		
		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			std::string line = "";
			std::vector<std::string> sline; sline.reserve(4);
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
				positions.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "vn") {
				normals.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "f") {
				std::vector<T2> f; f.reserve(3);

				for (size_t i = 1; i < sline.size(); i++) {
					auto left = sline[i].find('/');
					int pi, ni = -1;

					if (left == std::string::npos) {
						pi = stoi(sline[i]);

					} else {
						auto right = sline[i].rfind('/');
						auto s0 = sline[i].substr(0, left);
						auto s2 = sline[i].substr(right + 1, sline[i].length() - right);

						pi = stoi(s0);

						if (s2 != "") {
							ni = stoi(s2);
							ni = ni < 0 ? ni + normals.size() : ni - 1;
						}
					}
					pi = pi < 0 ? pi + positions.size() : pi - 1;

					if (unique_verts.count({ pi, ni }) != 0) {
						f.push_back(unique_verts[{ pi, ni }]);
					} else {
						unique_verts[{ pi, ni }] = index;
						f.push_back(index++);
						vertices.emplace_back(positions[pi]);
						v_normals.push_back(ni < 0 ? T1({ 0.0, 0.0, 0.0 }) : normals[ni]);
					}
				}
				faces.emplace_back(f);
			}
		}
		return { vertices, faces, v_normals };
	}

	template<typename T1 = std::vector<double>, typename T2 = std::vector<double>, typename T3 = int>
	std::tuple<std::vector<T1>,
		std::vector<std::vector<T3>>,
		std::vector<T2>> loadOBJvft(const std::string path) {

		std::vector<T1> vertices;
		std::vector<std::vector<T3>> faces;
		std::vector<T2> v_texcoords;

		std::vector<T1> positions;
		std::vector<T2> texcoords;
		std::map<std::pair<int, int>, size_t> unique_verts;
		size_t index = 0;

		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cerr << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			std::string line = "";
			std::vector<std::string> sline; sline.reserve(4);
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
				positions.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "vt") {
				texcoords.push_back({ stod(sline[1]), stod(sline[2]) });
			} else if (sline[0] == "f") {
				std::vector<T3> f; f.reserve(3);

				for (size_t i = 1; i < sline.size(); i++) {
					auto left = sline[i].find('/');
					int pi, ti = -1;

					if (left == std::string::npos) {
						pi = stoi(sline[i]);

					} else {
						auto right = sline[i].rfind('/');
						auto s0 = sline[i].substr(0, left);
						auto s1 = sline[i].substr(left + 1, right - left - 1);

						pi = stoi(s0);

						if (s1 != "") {
							ti = stoi(s1);
							ti = ti < 0 ? ti + texcoords.size() : ti - 1;
						}
					}
					pi = pi < 0 ? pi + positions.size() : pi - 1;

					if (unique_verts.count({ pi, ti }) != 0) {
						f.push_back(unique_verts[{ pi, ti }]);
					} else {
						unique_verts[{ pi, ti }] = index;
						f.push_back(index++);
						vertices.emplace_back(positions[pi]);
						v_texcoords.push_back(ti < 0 ? T2({ 0.0, 0.0 }) : texcoords[ti]);
					}
				}
				faces.emplace_back(f);
			}
		}

		return { vertices, faces, v_texcoords };
	}

	template<typename T1 = std::vector<double>, typename T2 = std::vector<double>, typename T3 = int>
	std::tuple<std::vector<T1>,
		std::vector<std::vector<T3>>,
		std::vector<T1>,
		std::vector<T2>> loadOBJvfnt(const std::string path) {

		std::vector<T1> vertices;
		std::vector<std::vector<T3>> faces;
		std::vector<T1> v_normals;
		std::vector<T2> v_texcoords;

		std::vector<T1> positions;
		std::vector<T1> normals;
		std::vector<T2> texcoords;
		std::map<std::tuple<int, int, int>, size_t> unique_verts;
		size_t index = 0;

		std::ifstream f(path, std::ios::ate);
		if (f.fail()) { std::cerr << "Failed to open file" << std::endl;  exit(1); };

		size_t fsize = (size_t)f.tellg();
		f.seekg(0);
		std::vector<char> text(fsize);

		f.read(text.data(), fsize);
		f.close();

		size_t seek = 0;
		while (seek < fsize) {
			std::string line = "";
			std::vector<std::string> sline; sline.reserve(4);
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
				positions.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "vt") {
				texcoords.push_back({ stod(sline[1]), stod(sline[2]) });
			} else if (sline[0] == "vn") {
				normals.push_back({ stod(sline[1]), stod(sline[2]), stod(sline[3]) });
			} else if (sline[0] == "f") {
				std::vector<T3> f; f.reserve(3);

				for (size_t i = 1; i < sline.size(); i++) {
					auto left = sline[i].find('/');
					int pi, ti = -1, ni = -1;

					if (left == std::string::npos) {
						pi = stoi(sline[i]);

					} else {
						auto right = sline[i].rfind('/');
						auto s0 = sline[i].substr(0, left);
						auto s1 = sline[i].substr(left + 1, right - left - 1);
						auto s2 = sline[i].substr(right + 1, sline[i].length() - right);

						pi = stoi(s0);

						if (s1 != "") {
							ti = stoi(s1);
							ti = ti < 0 ? ti + texcoords.size() : ti - 1;
						}
						if (s2 != "") {
							ni = stoi(s2);
							ni = ni < 0 ? ni + normals.size() : ni - 1;
						}
					}
					pi = pi < 0 ? pi + positions.size() : pi - 1;

					if (unique_verts.count({ pi, ti, ni }) != 0) {
						f.push_back(unique_verts[{ pi, ti, ni }]);
					} else {
						unique_verts[{ pi, ti, ni }] = index;
						f.push_back(index++);
						vertices.emplace_back(positions[pi]);
						v_texcoords.push_back(ti < 0 ? T2({ 0.0, 0.0 }) : texcoords[ti]);
						v_normals.push_back(ni < 0 ? T1({ 0.0, 0.0, 0.0 }) : normals[ni]);
					}
				}
				faces.emplace_back(f);
			}
		}

		return { vertices, faces, v_normals, v_texcoords };
	}

	template<typename T1, typename T2>
	void saveOBJ(std::string_view save_name, const T1& vertices, const T2& indices) {
		std::ofstream f(save_name.data());

		for (auto& v : vertices) {
			f << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
		}
		for (auto& idx : indices) {
			f << "f " << idx[0] + 1 << " " << idx[1] + 1 << " " << idx[2] + 1 << "\n";
		}
		f.close();
	}

	template<typename T1>
	static void saveVert(const std::string& file_name, std::vector<T1>& vertices) {
		std::ofstream f(file_name);
		std::string wr = "";

		for (int i = 0; i < vertices.size(); i++) {
			wr += std::string("v " + to_string(vertices[i][0]) + " " + to_string(vertices[i][1]) + " " + to_string(vertices[i][2]) + "\n");
		}

		f << wr;
		f.close();
	}

	template<typename T1>
	static void saveVertMP(const std::string& file_name, std::vector<T1>& vertices, const uint32_t num_threads = 8) {
		std::ofstream f(file_name);
		std::vector<std::string> wr(num_threads, "");

		const int th_num_verts = vertices.size() / num_threads;

		omp_set_num_threads(num_threads);
		#pragma omp parallel for
		for (int i = 0; i < num_threads; i++) {
			for (int j = th_num_verts * i; j < th_num_verts * (i + 1); j++) {
				wr[i] += to_string(vertices[j][0]) + " " + to_string(vertices[j][1]) + " " + to_string(vertices[j][2]) + "\n";
			}
		}

		for (int j = th_num_verts * num_threads; j < vertices.size(); j++) {
			wr[0] += to_string(vertices[j][0]) + " " + to_string(vertices[j][1]) + " " + to_string(vertices[j][2]) + "\n";
		}

		for (int i = 0; i < num_threads; i++) {
			f << wr[i];
		}

		f.close();
	}

	template<typename T1, typename T2>
	static void saveTriOBJ(const std::string& file_name, std::vector<T1>& vertices, std::vector<T2>& indices) {
		std::ofstream f(file_name);
		std::string wr = "";

		for (auto& v : vertices)
			wr += "v " + to_string(v[0]) + " " + to_string(v[1]) + " " + to_string(v[2]) + "\n";

		for (auto& idx : indices)
			wr += "f " + to_string(idx[0] + 1) + " " + to_string(idx[1] + 1) + " " + to_string(idx[2] + 1) + "\n";

		f << wr;
		f.close();
	}

};