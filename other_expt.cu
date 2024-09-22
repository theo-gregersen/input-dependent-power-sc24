#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <unistd.h>

// Flip the element's bits
__global__ void FlipBits_kernel(int8_t *A, int rows, int columns, uint64_t mask) {
    unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        unsigned long offset = i + j * rows;
        A[offset] = A[offset] ^ mask;
    }
}

cudaError_t AllocateMatrix(int8_t **matrix, int rows, int columns, std::string pattern) {
    size_t sizeof_matrix = sizeof(int8_t) * rows * columns;
    size_t rc = static_cast<unsigned long>(rows) * columns; // To avoid overflow
    int8_t *local_matrix = (int8_t *)calloc(rc, sizeof(int8_t));

    int8_t value = 0;
    if (pattern.compare("ones") == 0) {
        unsigned int hex = 0xFFFFFFFF;
        value = *(reinterpret_cast<int8_t *>(&hex));
    }

    for (unsigned long i = 0; i < rc; i++) {
        local_matrix[i] = value;
    }

    cudaError_t result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: "
            << cudaGetErrorString(result) << std::endl;
        return result;
    }

    cudaMemcpy(*matrix, local_matrix, sizeof_matrix, cudaMemcpyHostToDevice);

    free(local_matrix);

    return cudaGetLastError();
}

cudaError_t TestMemory(int rows, int columns, std::string pattern, std::string output_path, int n) {
    int8_t *matrix;
    cudaError_t result = AllocateMatrix(&matrix, rows, columns, pattern);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: "
            << cudaGetErrorString(result) << std::endl;
        return result;
    }

    int sys_output;
    std::string dcgm = "dcgmi dmon -e 100,101,112,140,150,155,156,190,191,203,204,206,207,210,211,240,241,242,243,244,245,246,247,252,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -d 100 > " + output_path + " &";
    sys_output = std::system(dcgm.c_str());

    sleep(n);

    sys_output = std::system("pkill -f dcgmi");

    cudaFree(matrix);

    return cudaSuccess;
}

cudaError_t TestBitFlips(int rows, int columns, std::string output_path, int iterations, int n) {
    int8_t *matrix;
    cudaError_t result = AllocateMatrix(&matrix, rows, columns, "zeros");

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: "
            << cudaGetErrorString(result) << std::endl;
        return result;
    }

    uint64_t mask = 0x00000000;
    if (n > 0) {
        mask = mask | 0b1;
    }
    for (int i = 1; i < n; i++) {
        mask = mask << 1;
        mask = mask | 0b1;
    }

    dim3 block(16, 16);
    dim3 grid(
        (rows + block.x - 1) / block.x,
        (columns + block.y - 1) / block.y
    );

    int sys_output;
    std::string dcgm = "dcgmi dmon -e 100,101,112,140,150,155,156,190,191,203,204,206,207,210,211,240,241,242,243,244,245,246,247,252,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -d 100 > " + output_path + " &";
    sys_output = std::system(dcgm.c_str());

    for (int i = 0; i < iterations; i++) {
        FlipBits_kernel<<< grid, block >>>(matrix, rows, columns, mask);
    }

    sys_output = std::system("pkill -f dcgmi");

    cudaFree(matrix);

    return cudaGetLastError();
}

int main(int argc, const char *arg[]) {

    int rows;
    std::stringstream r(arg[1]);
    r >> rows;

    int columns;
    std::stringstream c(arg[2]);
    c >> columns;

    std::string pattern;
    std::stringstream ss(arg[3]);
    ss >> pattern;

    std::string output_path;
    std::stringstream op(arg[4]);
    op >> output_path;

    int iterations;
    std::stringstream is(arg[5]);
    is >> iterations;

    int n = 0;
    std::stringstream ns(arg[6]);
    ns >> n;

    cudaError_t result;
    if (pattern.compare("bit-flips") == 0) {
        result = TestBitFlips(rows, columns, output_path, iterations, n);
    } else {
        result = TestMemory(rows, columns, pattern, output_path, n);
    }

    if (result == cudaSuccess) {
        std::cout << "Passed" << std::endl;
    } else {
        std::cout << cudaGetErrorString(result) << std::endl;
    }

    // Exit.
    return result == cudaSuccess ? 0 : -1;
}