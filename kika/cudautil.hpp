//// <<< grid, block >>>
//// num of target computation N -> grid * block = N
////
//// Maximum number of threads per block : 1024
//// Maximum sizes of each dimension of a block : 1024 x 1024 x 64 (X x Y x Z <= 1024, (33 x 32 x 1 is not legal))
//// Maximum sizes of each dimension of a grid : 2147483647 x 65535 x 65535
////
//// gridDim -> blockDim -> blockIdx -> threadIdx
////  const  ->   const  -> [0,grid] -> [0,block]
////-----------------------------------------------------
//// 1-dim :
//// uint32_t i = blockIdx.x * blockDim.x + threadIdx.x
//// 
//// for i:
//// for i (blockIdx) in range(gridDim = numBlock)
////    for j (threadIdx) in range(r = blockDim(numThread))
////
////-----------------------------------------------------
//// 2-dim :
//// uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
//// uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
//// uint32_t i = ix * (gridDim.y * blockDim.y) + iy;
//// 
//// for ix:
////   for iy:
////
////-----------------------------------------------------
//// 3-dim :
//// uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
//// uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
//// uint32_t iz = blockIdx.z * blockDim.z + threadIdx.z;
//// uint32_t i = ix * (gridDim.y * blockDim.y) * (gridDim.z * blockDim.z) + iy * (gridDim.z * blockDim.z) + iz;
//// 
//// for ix: for iy: for iz:

#pragma once
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace kika::cuda {

//
// std::thread setupCuda([]() { cudaFree(0); });
// setupCuda.join();
//

template <typename T, bool Pinned = true>
class CuData {
public:
	T* host;
	T* device;
	size_t size;

	CuData() : host(nullptr), device(nullptr), size(0) {}

	// @param size : size of allocation (byte)
	CuData(const size_t size) : CuData() {
		alloc(size);
	}

	~CuData() {
		free();
	}

	// @param size : size of allocation (byte)
	void alloc(const size_t size) {
		this->size = size;
		if constexpr (Pinned) {
			cudaMallocHost((void**)&host, size);
		} else {
			host = (T*)malloc(size);
		}
		cudaMalloc((void**)&device, size);
		assert(host != nullptr && device != nullptr);
	}

	void free() {
		cudaFree(device);
		device = nullptr;
		if constexpr (Pinned) cudaFreeHost(host);
		else free(host);
		host = nullptr;
		this->size = 0;
	}

	void realloc(const size_t size) {
		free();
		alloc(size);
	}

	void host2device() {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
	}

	void host2device(const size_t partialSize) {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy(device, host, partialSize, cudaMemcpyHostToDevice);
	}

	void host2device(const size_t offset, const size_t partialSize) {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy(device + offset, host + offset, partialSize, cudaMemcpyHostToDevice);
	}

	void device2host() {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
	}

	void host2deviceAsync(cudaStream_t& stream) {
		cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice, stream);
	}

	void device2hostAsync(cudaStream_t& stream) {
		cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, stream);
	}

	template <typename V>
	void set(const V* value) {
		cudaMemcpy(host, value, size, cudaMemcpyHostToHost);
		host2device();
	}
};

template <typename T, bool Pinned = true>
class CuData2D {
public:
	T* host;
	T* device;
	size_t size, dp, hp, width, height;

	CuData2D() : host(nullptr), device(nullptr), size(0), dp(0), hp(0), width(0), height(0) {}

	~CuData2D() {
		free();
	}

	// @param size : size of allocation (byte)
	void alloc(const size_t width, const size_t height) {
		hp = width;
		this->width = width;
		this->height = height;
		size = width * height;
		if constexpr (Pinned) cudaMallocHost((void**)&host, size);
		else host = (T*)malloc(size);
		cudaMallocPitch((void**)&device, &dp, width, height);
		assert(host != nullptr && device != nullptr);
	}

	void free() {
		if constexpr (Pinned) cudaFreeHost(host);
		else free(host);
		host = nullptr;
		cudaFree(device);
		device = nullptr;
		this->size = 0;
	}

	void realloc(const size_t width, const size_t height) {
		free();
		alloc(width, height);
	}

	void host2device() {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy2D(device, dp, host, hp, width, height, cudaMemcpyHostToDevice);
	}

	void host2device(const size_t partialWidth) {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy2D(device, dp, host, hp, partialWidth, height, cudaMemcpyHostToDevice);
	}

	void device2host() {
		assert(host != nullptr && device != nullptr);
		cudaMemcpy2D(host, hp, device, dp, width, height, cudaMemcpyHostToDevice);
	}
};

inline std::tuple<int, int, int> copySparseMatData(const Eigen::SparseMatrix<double>& A, 
	CuData<int>& rowIndex, CuData<int>& colIndex, CuData<double>& nnzValue) {
	auto rows = A.rows(), cols = A.cols();
	auto nnzs = A.nonZeros();

	rowIndex.alloc((rows + 1) * sizeof(int));
	colIndex.alloc(nnzs * sizeof(int));
	nnzValue.alloc(nnzs * sizeof(double));

	memcpy(rowIndex.host, A.outerIndexPtr(), rows * sizeof(int));
	rowIndex.host[rows] = nnzs;
	memcpy(colIndex.host, A.innerIndexPtr(), colIndex.size);
	memcpy(nnzValue.host, A.valuePtr(), nnzValue.size);

	return std::make_tuple((int)rows, (int)cols, (int)nnzs);

	// auto [row, col, nnz] = kika::eig::getCsrDataFromSparseMatUpper(A, 0, true);
	// 
	// auto rows = A.rows(), cols = A.cols();
	// auto nnzs = nnz.size();
	// 
	// rowIndex.alloc((rows + 1) * sizeof(int));
	// colIndex.alloc(nnzs * sizeof(int));
	// nnzValue.alloc(nnzs * sizeof(double));
	// 
	// memcpy(rowIndex.host, row.data(), rowIndex.size);
	// memcpy(colIndex.host, col.data(), colIndex.size);
	// memcpy(nnzValue.host, nnz.data(), nnzValue.size);
	// 
	// return std::make_tuple((int)rows, (int)cols, (int)nnzs);
}

}