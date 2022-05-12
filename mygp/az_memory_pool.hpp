#pragma once
//#include <cassert>
#include <algorithm>
#include <vector>

namespace gp::util {

// simple memory pool
template <class... MTypes>
class AZMemoryPool {
	std::vector<char*> memBuffers;
	size_t slot, offset, bufferSize;

	template <typename... Sizes>
	inline size_t accumulate(Sizes&&... sizes) { return (sizes + ...); }

	void exalloc(size_t size) {
		offset = 0;
		bufferSize = size;
		slot++;
		memBuffers.emplace_back((char*)malloc(bufferSize));
	}

public:

	AZMemoryPool() : memBuffers({ nullptr }), slot(0), offset(0), bufferSize(0) {
		std::vector<size_t> align{ alignof(MTypes)... };
		assert("AZMemoryPool" && !align.empty());
		assert("AZMemoryPool" && 
			std::all_of(align.begin()+1, align.end(), [&align](const size_t& i) { return i == align[0]; }));
	}

	~AZMemoryPool() { reset(); }

	/// init allocation
	template <typename... Sizes>
	void ialloc(Sizes... sizes) {
		assert("AZMemoryPool(allocate)" && (sizeof...(MTypes) == sizeof...(Sizes)));
		reset();
		bufferSize = accumulate(sizeof(MTypes) * static_cast<size_t>(sizes) ...);
		memBuffers[0] = (char*)malloc(bufferSize);
	}

	// bind memory
	template <typename T>
	char* bind() {
		if (offset + sizeof(T) > bufferSize)
			exalloc(offset+sizeof(T));

		offset += sizeof(T);
		return memBuffers[slot]+(offset-sizeof(T));
	}

	// reset resources
	void reset() {
		offset = 0;
		bufferSize = 0;
		for (size_t i = 0; i < slot+1; i++)
			free(memBuffers[i]);
		memBuffers.resize(1);
		memBuffers[0] = nullptr;
		slot = 0;
	}

};

class DynamicBitset {
	size_t size_;
	std::vector<uint64_t> data;

public:
	DynamicBitset(const size_t size) : size_(size), data(size / 64 + (size%64 != 0), 0uLL) {}

	void reset() { std::fill(data.begin(), data.end(), 0uLL); }

	/// bitset[i] = true; 
	void set(size_t i) { data[i/64] |= (1uLL << (i%64)); }

	bool operator[](size_t i) const { return 1 & (data[i/64] >> (i%64)); }

	size_t size() const { return size_; }
};

}