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


#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper_math.h"
#include <cmath>

#include <thread>
#include <iostream>
#include <numeric>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "for_cxx17.hpp"

// (32-bit unsigned only) Round up to the next higher power of 2.
// (exception) if v == 0 then return 0
unsigned int rupow2(unsigned int v) {
  if (v == 0) return 0;
  v--;
  for (unsigned int i = 1; i <= 16; i*=2) {
    v |= v >> i;
  }
  v++;
  return v;
}

__device__ inline float solidAngle(float3 a, float3 b, float3 c) {
  float r0 = length(a), r1 = length(b), r2 = length(c);
  float d = r0 * r1 * r2 + dot(a, b) * r2 + dot(b, c) * r0 + dot(c, a) * r1;
  return 2.0 * atan2(dot(a, cross(b, c)), d);
}

__global__ void computeFaceAreas(uint32_t* info, float3* tris, float* areas) {
  uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t iy = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t i = ix * (gridDim.y * blockDim.y) + iy;

  //extern __shared__ float3 shtris[];
  //if (i == 0) {
  //  for (int j = 0; j < 512*4; j++) { shtris[j] = tris[j]; }
  //}
  //__syncthreads();
  //
  //for (uint32_t t = 0; t < 512; t++) {
  //  angles[t] += 0.5 * length(cross(shtris[t*4+1] - shtris[t*4+0], shtris[t*4+2] - shtris[t*4+0]));
  //}
  //
  //for (uint32_t t = 512; t < info[0]; t++) {
  //  angles[t] += 0.5 * length(cross(tris[t*4+1] - tris[t*4+0], tris[t*4+2] - tris[t*4+0]));
  //}
  
  if (i >= info[0]) return;

  //__shared__ float3 stris[1];
  //stris[0] = tris[i*4+0];
  //areas[i] = 0.5 *length(cross(tris[i*4+0] - tris[i*4+1], tris[i*4+2] - tris[i*4+0]));
  areas[i] = 0.5 *length(cross(tris[i*4+1] - tris[i*4+0], tris[i*4+2] - tris[i*4+0]));

  //angles[i] = solidAngle(shtris[threadIdx.x*4+0], shtris[threadIdx.x*4+1], shtris[threadIdx.x*4+2]);
  //angles[i] = 0.5 * length(cross(shtris[threadIdx.x*4+1] - shtris[threadIdx.x*4+0], shtris[threadIdx.x*4+2] - shtris[threadIdx.x*4+0]));
  //angles[i] = 0.5 * length(cross(tris[i*4+1] - tris[i*4+0], tris[i*4+2] - tris[i*4+0]));
}

float reduce(float* data, uint32_t n) {
  float sum = 0.f;
  //for (uint32_t i = 1; i < *n; i*=2) {
  //  for (uint32_t j = 0; j < *n; j++) {
  //    if (j % (2*i) == 0) data[j] += data[j+i];
  //  }
  //}
  //sum = data[0];

  for (uint32_t i = 0; i < n; i++) {
    sum += data[i];
  }

  return sum;
}

//// dim3 block(th, 1, 1);
//// dim3 grid(rupow2(n/th), 1, 1);
//// dim3 subgrid(rupow2((unsigned int)std::ceil((grid.x-1)/1024.)), 1, 1);
//// dim3 subblock(std::min(1024u, grid.x), 1, 1);
//// dim3 subsubgrid(rupow2((unsigned int)std::ceil((subgrid.x-1)/1024.)), 1, 1);
//// dim3 subsubblock(std::min(1024u, subgrid.x), 1, 1);
// 
//// dev_reduction<<<grid.x, block.x, block.x*sizeof(float)>>>(areas.device, result.device, *nf.host);
//// if (subgrid.x) dev_reduction<<<subgrid.x, subblock.x, subblock.x*sizeof(float)>>>(result.device, result.device+grid.x, grid.x);
//// if (subsubgrid.x) dev_reduction<<<subsubgrid.x, subsubblock.x, subsubblock.x*sizeof(float)>>>(result.device+grid.x, result.device+grid.x+subgrid.x, subgrid.x);
////
////__global__ void dev_reduction(float* data, float* sum, uint32_t n) {
////   extern __shared__ float sdata[];
////   
////   auto tid = threadIdx.x;
////   auto i = blockIdx.x*blockDim.x + threadIdx.x;
////   if (i >= n) return;
////   sdata[tid] = data[i] + data[i];
////   __syncthreads();
////   
////   for (unsigned int s=blockDim.x/2; s > 0; s>>=1) {
////     if (tid < s) {
////       sdata[tid] += sdata[tid + s];
////     }
////     __syncthreads();
////   }
////   
////   if (tid == 0) sum[blockIdx.x] = sdata[0];
//// }

__global__ void cureduceKernel(float* data, float* sum, uint32_t n) {
  extern __shared__ float sdata[];

  auto tid = threadIdx.x;
  auto i = blockIdx.x*blockDim.x*2 + threadIdx.x;
  sdata[tid] = (i < n) ? (data[i] + ((i + blockDim.x < n) ? data[i + blockDim.x] : 0.f)) : 0.f;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s > 0; s>>=1) {
    sdata[tid] += (tid < s) ? sdata[tid+s] : 0;
    __syncthreads();
  }

  if (tid == 0) sum[blockIdx.x] = sdata[0];
}

template <typename T>
concept Req = requires(T t) {
  t.size;
};
void function(T a) {
  std::cout << typeid(a).name() << std::endl;
}

// 1d array reduction sum (max size : n <= 2^30)
// @param d_data : array data on device
// @param n : 1d array size
// @param th : num threads per thread-blcok
float cureduce(float* d_data, const uint32_t n, const uint32_t th = 512) {
  dim3 block1(th, 1, 1);
  dim3 grid1(rupow2((unsigned int)std::ceil(n/(block1.x*2.))), 1, 1);

  dim3 block2(std::min(block1.x, grid1.x), 1, 1);
  dim3 grid2(rupow2((unsigned int)std::ceil((grid1.x-1.)/(block1.x*2.))), 1, 1);

  dim3 block3(std::min(block1.x, grid2.x), 1, 1);
  dim3 grid3(rupow2((unsigned int)std::ceil((grid2.x-1.)/(block2.x*2.))), 1, 1);
  
  dim3 block4(std::min(block1.x, grid3.x), 1, 1);
  dim3 grid4(rupow2((unsigned int)std::ceil((grid3.x-1.)/(block3.x*2.))), 1, 1);

  //std::cout << grid1.x << " " << grid2.x << " " << grid3.x << " " << grid4.x << std::endl;
  //std::cout << block1.x << " " << block2.x << " " << block3.x << " " << block4.x << std::endl;

  gp::cuda::CuData<float> result(sizeof(float)*(grid1.x+grid2.x+grid3.x+grid4.x));
  cudaMemset(result.device, 0, result.size);

  cureduceKernel<<<grid1.x, block1.x, block1.x*sizeof(float)>>>(d_data, result.device, n);
  if (grid2.x) cureduceKernel<<<grid2.x, block2.x, block2.x*sizeof(float)>>>(result.device, result.device+grid1.x, grid1.x);
  if (grid3.x) cureduceKernel<<<grid3.x, block3.x, block3.x*sizeof(float)>>>(result.device+grid1.x, result.device+grid1.x+grid2.x, grid2.x);
  if (grid4.x) cureduceKernel<<<grid4.x, block4.x, block4.x*sizeof(float)>>>(result.device+grid1.x+grid2.x, result.device+grid1.x+grid2.x+grid3.x, grid3.x);
  result.device2host();

  return result.host[grid1.x+grid2.x+grid3.x+grid4.x-1];
}

int main(int argc, char* argv[]) {
  Eigen::setNbThreads(1);
  Eigen::initParallel();
  
  std::thread setupCuda([]() { cudaFree(0); });
  //setupCuda.join();

  /*
  // n <= 2^30
  gp::cuda::CuData<uint32_t> nf(sizeof(uint32_t));
  *nf.host = pow(2, 30);
  std::cout << *nf.host << std::endl;

  gp::cuda::CuData<float> areas(sizeof(float) * *nf.host);

  {
    std::random_device seed;
    std::mt19937 engine(0);
    std::uniform_real_distribution<float> distrib(0.f, 1.f);

    for (uint32_t i = 0; i < *nf.host; i++) {
      areas.host[i] = 1.0;
    }
    std::cout.precision(16);
  }

  nf.host2device();
  areas.host2device();

  {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << reduce(areas.host, *nf.host) << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  }

  {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << cureduce(areas.device, *nf.host) << std::endl;
   
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  }
  */

  
  auto pf = gp::mio::loadOBJvf(argv[1]);
  auto V = gp::eig::vec2eig(std::get<0>(pf));
  auto F = gp::eig::vec2eig(std::get<1>(pf));
  auto N = gp::util::faceNormals(V, F);
  
  function(V);

  return 0;

  size_t nf = F.rows();
  setupCuda.join();

  dim3 block(1024, 1, 1);
  dim3 grid(512, 1, 1);

  gp::cuda::CuData<uint32_t> info;
  gp::cuda::CuData<float3> tris;
  gp::cuda::CuData<float> d_areas;

  info.alloc(sizeof(uint32_t));
  tris.alloc(sizeof(float3) * 4 * nf);
  d_areas.alloc(sizeof(float) * nf);
  Eigen::VectorXf areas(nf);

  *info.host =nf;
  for (size_t i = 0; i < nf; i++) {
    Eigen::Matrix3f tri = V(F.row(i), Eigen::all).transpose().cast<float>();
    Eigen::Vector3f vn = N.row(i).cast<float>();
    memcpy(tris.host + i * 4 + 0, tri.data(), sizeof(float3)*3);
    memcpy(tris.host + i * 4 + 3, vn.data(), sizeof(float3));
    areas[i] = 0.5*(tri.col(1) - tri.col(0)).cross(tri.col(2) - tri.col(0)).norm();
  }

  float h_surfaceArea = 0.f;
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nf; i++) {
      h_surfaceArea += 0.5*length(cross((tris.host[i*4+1] - tris.host[i*4+0]), (tris.host[i*4+2] - tris.host[i*4+0])));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
  }

  cudaMemset(d_areas.device, 0, d_areas.size);
  info.host2device();
  tris.host2device();

  {
    auto start = std::chrono::high_resolution_clock::now();

    computeFaceAreas<<<grid, block>>>(info.device, tris.device, d_areas.device);

    d_areas.device2host();

    float d_surfaceArea = 0.f;
    for (size_t i = 0; i < nf; i++) {
      d_surfaceArea += d_areas.host[i];
    }
    //float d_surfaceArea = cureduce(d_areas.device, nf);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

    std::cout << d_surfaceArea << std::endl;
  }

  std::cout << d_areas.host[0] << " " << d_areas.host[1] << " " << d_areas.host[2] << std::endl;
  std::cout << h_surfaceArea << std::endl;
  std::cout << areas[0] << " " << areas[1] << " " << areas[2] << std::endl;
  
}