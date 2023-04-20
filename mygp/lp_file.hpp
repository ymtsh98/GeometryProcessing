#pragma once

#include <iostream>
#include <string>
#include <filesystem>

namespace gp::util {

template <char C, char... Args>
inline constexpr bool chareq(const char& c) {
	if constexpr (sizeof...(Args) == 0) {
		return (c == C);
	} else {
		return (c == C) || chareq<Args...>(c);
	}
}

/// @brief file loader/perser
class LPFile {

	size_t lenline() {
		size_t n = seek;
		while (n < size && !chareq<'\n'>(buffer[n])) {
			n++;
		}
		return n - seek;
	}

	size_t lenseg() {
		size_t n = seek;
		while (n < size && !chareq<' ', '\n', '	', '\0'>(buffer[n])) {
			n++;
		}
		return n - seek;
	}

	size_t lenspace() {
		size_t n = seek;
		while (n < size && chareq<' ', '	', '\0'>(buffer[n])) {
			n++;
		}
		return n - seek;
	}

public:
	size_t seek, size;
	std::unique_ptr<char[]> buffer;
	std::string line;

	LPFile() : seek(0), size(0), buffer(nullptr), line({}) {
		line.reserve(256);
	}

	LPFile(std::string_view path) : seek(0), size(0), buffer(nullptr), line({}) {
		open(path);
		line.reserve(256);
	}

	void open(std::string_view path) {
		std::ifstream file(path, std::ios::in);
		if (file.fail()) {
			std::cerr << "Failed to open file : " << path << std::endl;
			exit(1);
		}

		size = (size_t)std::filesystem::file_size(path.data());
		buffer = std::make_unique<char[]>(size);

		if (buffer != nullptr) {
			file.read(buffer.get(), size);
		} else {
			std::cerr << "Failed to load file : " << path << std::endl;
			exit(1);
		}

		file.close();
	}

	void __debug(std::string_view data) {
		seek = 0;
		size = data.size();
		buffer = std::make_unique<char[]>(size);
		if (buffer != nullptr)
			for (size_t i = 0; i < size; i++) buffer[i] = data[i];
	}

	void skipline() {
		seek += lenline()+1;
	}

	void skipseg() {
		seek += lenspace();
		seek += lenseg()+1;
	}

	std::string& getline() {
		line.clear();
		while (seek < size && !chareq<'\n'>(buffer[seek])) {
			line += buffer[seek++];
		}
		seek++;
		return line;
	}

	std::string& getseg() {
		line.clear();
		seek += lenspace();
		while (seek < size && !chareq<' ', '\n', '	'>(buffer[seek])) {
			line += buffer[seek++];
		}
		seek += lenspace();
		seek += chareq<'\n'>(buffer[seek]);
		return line;
	}

	std::string& getseg(bool& endl) {
		endl = false;
		line.clear();
		seek += lenspace();
		while (seek < size && !chareq<' ', '\n', '	'>(buffer[seek])) {
			line += buffer[seek++];
		}
		seek += lenspace();
		if (eof() || chareq<'\n'>(buffer[seek])) {
			seek++;
			endl = true;
		}
		return line;
	}

	const char operator()() const {
		return buffer[seek];
	}

	bool eof() const {
		return seek >= size;
	}

	std::string& curtline() {
		size_t n = 0;
		line.clear();
		while ((seek+n) < size && !chareq<'\n'>(buffer[seek+n])) {
			line += buffer[seek+n++];
		}
		return line;
	}

	std::string& curtseg() {
		size_t n = 0;
		line.clear();
		n += lenspace();
		while ((seek+n) < size && !chareq<' ', '\n', '	'>(buffer[seek+n])) {
			line += buffer[seek+n++];
		}
		return line;

	}

};
}