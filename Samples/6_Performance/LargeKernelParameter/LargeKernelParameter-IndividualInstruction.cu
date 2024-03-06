/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This is a simple test showing performance and usability
 * improvements with large kernel parameters introduced in CUDA 12.1
 */
#include <chrono>
#include <iostream>
#include <cassert>

// Utility includes
#include <helper_cuda.h>

using namespace std;
using namespace std::chrono;

//#define TEST_ITERATIONS     (1000)
//#define TOTAL_PARAMS        (8188)  // ints
//#define KERNEL_PARAM_LIMIT  (8188)  // ints
//#define CONSTANT_PARAM_LIMIT (16384)  // ints
//#define CONST_COPIED_PARAMS (CONSTANT_PARAM_LIMIT - 0)

#define TEST_ITERATIONS     (1)
#define TOTAL_PARAMS        (8180)  // ints
#define KERNEL_PARAM_LIMIT  (8180)  // ints //5460
#define CONSTANT_PARAM_LIMIT (16384)  // ints //4680
#define CONST_COPIED_PARAMS (CONSTANT_PARAM_LIMIT)
#define STRIDE (4)
#define NUMWAVES (1)
#define WARPSIZE (32)

typedef int IDT;

__constant__ int excess_params[CONST_COPIED_PARAMS];

#define SM_SIZE 0x1
__shared__ int sharedData[SM_SIZE];

typedef struct {
  IDT param[KERNEL_PARAM_LIMIT];
} param_t;

typedef struct {
  IDT param[TOTAL_PARAMS];
} param_large_t;

// Kernel with 4KB kernel parameter limit
__global__ void kernelDefault(__grid_constant__ const param_t p, int *result) {
  int tid = (threadIdx.x & 0x1F);
  int wid = (threadIdx.x >> 5);
  IDT tmp = 0;
  volatile int start_time, end_time;

  if ( tid & 0xF == 0) {
    //printf("%d\n", wid);
    sharedData[0] = clock();
    start_time = clock();
    // accumulate kernel parameters
  #pragma unroll
    for (int i = 0; i < KERNEL_PARAM_LIMIT; i+=WARPSIZE) {
      tmp += p.param[i+tid];
      tmp += sharedData[i & (SM_SIZE-1)];
    }
    // accumulate excess values passed via const memory
  #pragma unroll
    for (int i = 0; i < CONST_COPIED_PARAMS; i+=WARPSIZE) {
      tmp += excess_params[i+tid];
      tmp += sharedData[i & (SM_SIZE-1)];
    }

    end_time = clock();

  }
  if (tid == 0) {
    result[wid*STRIDE] = tmp;
    result[wid*STRIDE + 1] = start_time;
    result[wid*STRIDE + 2] = end_time;
  }
}

// Kernel with 32,764 byte kernel parameter limit
__global__ void kernelLargeParam(__grid_constant__ const param_large_t p, int *result) {
  int tid = (threadIdx.x & 0x1F);
  int wid = (threadIdx.x >> 5);
  IDT tmp = 0;

  volatile int start_time, end_time;
  if ( tid < 8) {
    sharedData[0] = clock();
    // accumulate kernel parameters
    if (wid == 0) {
      //printf("%d\n", wid);
      start_time = clock();
#pragma unroll
      //for (int i = 0; i < TOTAL_PARAMS; i += WARPSIZE) {
      for (int i = 0; i < 128; i += WARPSIZE) {
        tmp += p.param[i + tid];
        tmp += sharedData[i & (SM_SIZE - 1)];
      }

      end_time = clock();
      /*
    } else {
      //printf("%d\n", wid);
      start_time = clock();
#pragma unroll
      for (int i = 0; i < TOTAL_PARAMS; i += WARPSIZE) {
        tmp += p.param[i+1+tid];
        tmp += sharedData[i & (SM_SIZE-1)];
      }
      end_time = clock();
    }
    */
    }
      int t = clock();
  if (tid == 0) {
    result[wid * STRIDE] = tmp + t;
    result[wid*STRIDE + 1] = start_time;
    result[wid*STRIDE + 2] = end_time;
  }
  }

}

static void report_time(std::chrono::time_point<std::chrono::steady_clock> start,
                        std::chrono::time_point<std::chrono::steady_clock> end,
                        int iters) {
  auto usecs = duration_cast<duration<float,
                                      microseconds::period>>(end - start);
  cout << usecs.count() / iters << endl;
}

int main() {
  int rc;
  cudaFree(0);

  param_t p;
  param_large_t p_large;

  // pageable host memory that holds excess constants passed via constant memory
  int *copied_params = (int *)malloc(CONST_COPIED_PARAMS * sizeof(IDT));
  assert(copied_params);

  // storage for computed result
  constexpr int WAVESIZE = NUMWAVES*WARPSIZE;
  constexpr int NOUT = NUMWAVES * STRIDE;

  int *d_result;
  int h_result[NOUT];
  checkCudaErrors(cudaMalloc(&d_result, NOUT*sizeof(int)));

  int expected_result = 0;

  // fill in data for validation
  for (int i = 0; i < KERNEL_PARAM_LIMIT; ++i) {
    p.param[i] = (i & 0xFF);
  }
  for (int i = KERNEL_PARAM_LIMIT; i < TOTAL_PARAMS; ++i) {
    copied_params[i - KERNEL_PARAM_LIMIT] = (i & 0xFF);
  }
  for (int i = 0; i < TOTAL_PARAMS; ++i) {
    p_large.param[i] = (i & 0xFF);
    expected_result += (i & 0xFF);
  }

  kernelLargeParam<<<1,WAVESIZE>>>(p_large, d_result);
  checkCudaErrors(cudaMemcpy(&h_result, d_result, NOUT*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());

  for (int i = 0; i < NUMWAVES; ++i) {
    int offset = i * STRIDE;
    printf("W:[%d] Large Executed Cycles: [%d - %d], [%d]\n", i, h_result[offset + 1], h_result[offset + 2], h_result[offset + 2] - h_result[offset + 1]);
  }

  if(h_result[0] != expected_result) {
    std::cout << "Test failed" << std::endl;
	 rc=-1;
	 goto Exit;    
  }

  // warmup, verify correctness
  checkCudaErrors(cudaMemcpyToSymbol(excess_params, copied_params, CONST_COPIED_PARAMS * sizeof(IDT), 0, cudaMemcpyHostToDevice));
  kernelDefault<<<1,WAVESIZE>>>(p, d_result);
  checkCudaErrors(cudaMemcpy(&h_result, d_result, NOUT*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());

  for (int i = 0; i < NUMWAVES; ++i) {
    int offset = i * STRIDE;
    printf("W:[%d] Large Executed Cycles: [%d - %d], [%d]\n", i, h_result[offset + 1], h_result[offset + 2], h_result[offset + 2] - h_result[offset + 1]);
  }

  if(h_result[0] != expected_result) {
    std::cout << "Test failed" << std::endl;
	 rc=-1;
	 goto Exit;    
  }


  // benchmark default kernel parameter limit
  {
    auto start = steady_clock::now();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
      checkCudaErrors(cudaMemcpyToSymbol(excess_params, copied_params, CONST_COPIED_PARAMS * sizeof(IDT), 0, cudaMemcpyHostToDevice));
      kernelDefault<<<1, 1>>>(p, d_result);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = steady_clock::now();
    std::cout << "Kernel 4KB parameter limit - time (us):";
    report_time(start, end, TEST_ITERATIONS);

    // benchmark large kernel parameter limit
    start = steady_clock::now();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
      kernelLargeParam<<<1, 1>>>(p_large, d_result);
    }  
    checkCudaErrors(cudaDeviceSynchronize());
    end = steady_clock::now();
    std::cout << "Kernel 32,764 byte parameter limit - time (us):";
    report_time(start, end, TEST_ITERATIONS);
  }
  std::cout << "Test passed!" << std::endl;
  rc=0;
Exit:
  // cleanup
  cudaFree(d_result);
  free(copied_params);
  return rc;
}
