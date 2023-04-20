#pragma once
#include <vector>
#include <intrin.h>

/// 
/// Dynamic Bitset (Fast Filling)
/// 
namespace kika::util {
template<typename T=uint8_t, size_t s = sizeof(T) * 8>
class DynamicBitset {
	size_t _size;
	std::vector<T> data;

public:
	DynamicBitset(const size_t size) : _size(size), data(size / s + (size%s != 0), static_cast<T>(0)) {}

	void reset() { std::fill(data.begin(), data.end(), static_cast<T>(0)); }

	// bitset[i] = true; 
	void set(size_t i) { data[i/s] |= (1uLL << (i%s)); }

	bool operator[](size_t i) const { return data[i/s] & (1uLL << (i%s)); }

	size_t size() const { return _size; }
};

/*
class DynamicBitsetSIMD {
	size_t _size;

	union Data {
		std::vector<__m256i> m;
		std::vector<uint8_t> u;

		Data(const size_t& s, const __m256i& i) : m(s, i) {}
		~Data() {}
	} data;

public:
	DynamicBitsetSIMD(const size_t size) : _size(size), data(size / 256 + (size%256 != 0), _mm256_setzero_si256()) {}

	void reset() { std::fill(data.m.begin(), data.m.end(), _mm256_setzero_si256()); }

	//// low performance
	//void set(size_t i) { 
	//	size_t j = i / 64 % 4, k = 1uLL << (i % 64);
	//	data[i / 256] = _mm256_or_si256(data[i / 256],
	//		_mm256_set_epi64x(
	//		(j != 3) ? 0 : k, 
	//		(j != 2) ? 0 : k, 
	//		(j != 1) ? 0 : k, 
	//		(j != 0) ? 0 : k));
	//}
	//
	//
	//bool operator[](size_t i) const { 
	//	size_t j = i / 64 % 4, k = 1uLL << (i % 64);
	//	return _mm256_testc_si256(data[i / 256],
	//		_mm256_set_epi64x(
	//		(j != 3) ? 0 : k, 
	//		(j != 2) ? 0 : k, 
	//		(j != 1) ? 0 : k, 
	//		(j != 0) ? 0 : k));
	//}

	// bitset[i] = true; 
	void set(size_t i) { data.u[i/8] |= (1uLL << (i%8)); }

	bool operator[](size_t i) const { return data.u[i/8] & (1uLL << (i%8)); }

	size_t size() const { return _size; }
};
*/

////void testDynamicBitset() {
////	std::random_device seed;
////	std::mt19937 engine(seed());
////
////	size_t size = 1000;
////
////	std::vector<bool> bvec(size, false);
////	kika::util::DynamicBitsetSIMD bset(size);
////
////	std::vector<uint32_t> rand(1000);
////	for (size_t i = 0; i < rand.size(); i++) {
////		rand[i] = engine() % size;
////		bvec[rand[i]] = true;
////		bset.set(rand[i]);
////	}
////
////	for (size_t i = 0; i < size; i++) {
////		if (bvec[i] != bset[i]) {
////			std::cout << "NG : DynamicBitset, " << i << " " << bvec[i] << " " << bset[i] << std::endl;
////			return;
////		}
////	}
////
////	std::cout << "OK : DynamicBitset" << std::endl;
////}
}