#pragma once
#include <cassert>
#include <algorithm>
#include <vector>

/// 
/// Memory Pool
/// 
namespace kika::util {

// simple memory pool
//
// usage: 
// AZMemoryPool<int> azmPool;
// azmPool.ialloc(100);
// auto p = azmPool.bind<int>();
template <typename MType, typename... MTypes>
class AZMemoryPool {
	std::vector<char*> memBuffers;
	size_t slot, offset, bufferSize;

	void exalloc(size_t size) {
		offset = 0;
		bufferSize = size;
		slot++;
		memBuffers.emplace_back((char*)malloc(bufferSize));
	}

public:

	AZMemoryPool() : memBuffers({ nullptr }), slot(0), offset(0), bufferSize(0) {
		static_assert(((alignof(MType) == alignof(MTypes)) && ... && true), "AZMemoryPool requires template arguments are the same align.");
	}

	AZMemoryPool(AZMemoryPool&& mpool) noexcept : 
		memBuffers(std::move(mpool.memBuffers)),
		slot(mpool.slot), offset(mpool.offset), bufferSize(mpool.bufferSize){
		mpool.slot = 0;
		mpool.memBuffers.resize(1);
		mpool.memBuffers[0] = nullptr;
	}

	~AZMemoryPool() { 
		reset(); 
	}

	/// init allocation
	template <typename Size, typename... Sizes>
	void ialloc(Size size, Sizes... sizes) {
		static_assert(sizeof...(MTypes) == sizeof...(Sizes), "AZMemoryPool(ialloc) : Sizes does not match MTypes.");
		reset();
		bufferSize = ((sizeof(MTypes) * static_cast<size_t>(sizes)) + ... + (sizeof(MType) * static_cast<size_t>(size)));
		memBuffers[0] = (char*)malloc(bufferSize);
	}

	// bind memory
	template <typename T>
	[[nodiscard]] char* bind() {
		static_assert((std::is_same_v<T, MTypes> || ... || std::is_same_v<T, MType>), "AZMemoryPool(bind) : T does not match MTypes.");
		if (offset + sizeof(T) > bufferSize) exalloc(offset+sizeof(T)); // memBuffers.back() size * 2 memory allocation 
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
}