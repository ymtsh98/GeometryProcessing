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

	CuData(const size_t size) : CuData() {
		alloc(size);
	}

	~CuData() {
		cudaFree(device);
		device = nullptr;
		if constexpr (Pinned) cudaFreeHost(host);
		else free(host);
		host = nullptr;
	}

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
};

template <typename T, bool Pinned = true>
class CuData2D {
public:
	T* host;
	T* device;
	size_t size, dp, hp, width, height;

	CuData2D() : host(nullptr), device(nullptr), size(0), dp(0), hp(0), width(0), height(0) {}

	~CuData2D() {
		if constexpr (Pinned) cudaFreeHost(host);
		else free(host);
		host = nullptr;
		cudaFree(device);
		device = nullptr;
	}

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

}