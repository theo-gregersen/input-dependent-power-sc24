/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/matrices.h"
#include "cutlass/numeric_conversion.h"
// #include "cutlass/numeric_types.h"

cutlass::NumericConverter<DTYPE, float> convert_float;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  DTYPE const *A,
  int lda,
  DTYPE const *B,
  int ldb,
  float beta,
  DTYPE *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  // Use these for tensor cores
  // using MMAOp = cutlass::arch::OpClassTensorOp;
  // using SmArch = cutlass::arch::Sm80;

  using CutlassGemm = cutlass::gemm::device::Gemm<DTYPE,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  DTYPE,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  DTYPE,        // Data-type of C matrix
                                                  ColumnMajor  // Layout of C matrix
                                                  >;

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {convert_float(alpha), convert_float(beta)}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Simple function to initialize a matrix.
cudaError_t InitializeMatrix(DTYPE *matrix, int rows, int columns, int seed, bool transpose, std::string pattern, float *params) {
  size_t matrix_size = rows * columns;
  float *local_matrix = (float *)calloc(matrix_size, sizeof(float));

  if (pattern.compare("gaussian") == 0) {
    MatrixGaussian(local_matrix, rows, columns, seed, params);
  } else if (pattern.compare("uniform") == 0) {
    MatrixUniform(local_matrix, rows, columns, seed, params);
  } else if (pattern.compare("unique") == 0) {
    MatrixGaussianUnique(local_matrix, rows, columns, seed, params);
  } else if (pattern.compare("bits") == 0) {
    MatrixInitialValue(local_matrix, rows, columns, seed, params);
  } else if (pattern.compare("bits_most") == 0) {
    MatrixInitialValue(local_matrix, rows, columns, seed, params);
  } else if (pattern.compare("bits_least") == 0) {
    MatrixInitialValue(local_matrix, rows, columns, seed, params);
  } else {
    MatrixGaussian(local_matrix, rows, columns, seed, params);
    float param = params[2];
    if (pattern.compare("sort_rows") == 0) {
      MatrixSortRows(local_matrix, rows, columns, param);
    } else if (pattern.compare("sort_columns") == 0) {
      MatrixSortColumns(local_matrix, rows, columns, param);
    } else if (pattern.compare("sort_rows_intra") == 0) {
      MatrixIntraSortRows(local_matrix, rows, columns, param);
    } else if (pattern.compare("sparsity") == 0) {
      MatrixAddSparsity(local_matrix, rows, columns, seed, param);
    } else if (pattern.compare("sort+sparsity") == 0) {
      MatrixSortRows(local_matrix, rows, columns, 1.0);
      MatrixAddSparsity(local_matrix, rows, columns, seed, param);
    }
  }

  if (transpose) {
    MatrixTranspose(local_matrix, rows, columns);
  }

  DTYPE *converted_matrix = (DTYPE *)calloc(matrix_size, sizeof(DTYPE));
  for (int i = 0; i < matrix_size; i++) {
    converted_matrix[i] = convert_float(local_matrix[i]);
  }
  matrix_size = sizeof(DTYPE) * matrix_size;

  // Perform bit manipulation after converting type
  if (pattern.compare("bits") == 0) {
    MatrixBits(converted_matrix, rows, columns, seed, params[2]);
  } else if (pattern.compare("bits_most") == 0) {
    MatrixBitsMost(converted_matrix, rows, columns, seed, params[2]);
  } else if (pattern.compare("bits_least") == 0) {
    MatrixBitsLeast(converted_matrix, rows, columns, seed, params[2]);
  } else if (pattern.compare("bits_most_zeros") == 0) {
    MatrixSetBitsMostZero(converted_matrix, rows, columns, params[2]);
  } else if (pattern.compare("bits_least_zeros") == 0) {
    MatrixSetBitsLeastZero(converted_matrix, rows, columns, params[2]);
  }

  cudaMemcpy(matrix, converted_matrix, matrix_size, cudaMemcpyHostToDevice);
  free(local_matrix);
  free(converted_matrix);

  // PrintMatrix(local_matrix, rows, columns, "./matrix.txt");
  // PrintMatrixBits(converted_matrix, rows, columns, "./matrix.txt");

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(DTYPE **matrix, int rows, int columns, int seed, bool tranpose, std::string pattern, float *params) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(DTYPE) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed, tranpose, pattern, params);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

// Allocate an empty matrix
cudaError_t AllocateMatrixEmpty(DTYPE **matrix, int rows, int columns) {
  cudaError_t result;
  size_t sizeof_matrix = rows * columns;
  sizeof_matrix = sizeof(DTYPE) * sizeof_matrix;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta,
  int iterations, int seed, std::string pattern, float *params, std::string output_path) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Define pointers to matrices in GPU device memory.
  DTYPE *A;
  DTYPE *B;
  DTYPE *C_cutlass;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, seed, false, pattern, params);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, seed+1, true, pattern, params);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrixEmpty(&C_cutlass, M, N);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  int sys_output;
  std::string dcgm = "dcgmi dmon -e 100,101,112,140,150,155,156,190,191,203,204,206,207,210,211,240,241,242,243,244,245,246,247,252,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -d 100 > " + output_path + " &";
  sys_output = std::system(dcgm.c_str());

  std::vector<long int> times;
  for (int i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    times.push_back(duration.count());

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }
  }

  sys_output = std::system("pkill -f dcgmi");

  float mean = static_cast<float>(std::accumulate(times.begin(), times.end(), 0.0)) / static_cast<float>(iterations);
  float max = static_cast<float>(*std::max_element(times.begin(), times.end()));
  printf("%f, %f, %d\n", mean / 1000.0, max / 1000.0, sys_output);

  //
  // Free device memory allocations.
  //

  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {
  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  // Similarity test parameters
  int iterations = 1;
  if (argc > 5) {
    std::stringstream ss(arg[6]);
    ss >> iterations;
  }

  int seed = 1;
  if (argc > 6) {
    std::stringstream ss(arg[7]);
    ss >> seed;
  }

  std::string pattern = "gaussian";
  if (argc > 7) {
    std::stringstream ss(arg[8]);
    ss >> pattern;
  }

  std::string output_path = "./dcgmi_output.csv";
  if (argc > 8) {
    std::stringstream ss(arg[9]);
    ss >> output_path;
  }

  float params[5] = { 0.0, 1.0 };
  for (int i = 10; i < argc; i++) {
    std::stringstream ss(arg[i]);
    ss >> params[i - 10];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],      // beta
    iterations,     // iterations
    seed,           // seed
    pattern,        // pattern
    params,         // additional params
    output_path      // output path for printing results
  );

  if (result == cudaSuccess) {
    std::cout << "Passed" << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
