#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>
#include <algorithm>

#include "cutlass/numeric_types.h"

#pragma once

std::default_random_engine DRE;
using DTYPE = cutlass::half_t; // int8_t, cutlass::half_t, float
size_t BIT_COUNT = sizeof(DTYPE) * 8;

void PrintMatrix(float *matrix, int rows, int columns, std::string filename) {
    FILE *f = fopen(filename.c_str(), "a");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int offset = i * columns + j;
            fprintf(f, "%.4f,", matrix[offset]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
    fclose(f);
}

std::string GetBitString(DTYPE value) {
    DTYPE value_copy = value;
    unsigned long *l = reinterpret_cast<unsigned long *>(&value_copy);
    std::string bits = "";

    for (size_t i = BIT_COUNT; i > 0; i--) {
        if (*l & 1) {
            bits = "1" + bits;
        } else {
            bits = "0" + bits;
        }
        *l = (*l >> 1);
    }
    return bits;
}

void PrintMatrixBits(DTYPE *matrix, int rows, int columns, std::string filename) {
    FILE *f = fopen(filename.c_str(), "a");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int offset = i * columns + j;
            fprintf(f, "%s,", GetBitString(matrix[offset]).c_str());
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
    fclose(f);
}

int GetHammingWeight(DTYPE value) {
    unsigned long *l = reinterpret_cast<unsigned long *>(&value);
    int weight = 0;

    for (size_t i = 0; i < BIT_COUNT; i++) {
        if ((*l >> i) & 1) {
            weight = weight + 1;
        }
    }

    return weight;
}

float GetBitAlignment(DTYPE value1, DTYPE value2) {
    unsigned long *l1 = reinterpret_cast<unsigned long *>(&value1);
    unsigned long *l2 = reinterpret_cast<unsigned long *>(&value2);
    float matches = 0;

    for (size_t i = 0; i < BIT_COUNT; i++) {
        if (((*l1 >> i) & 1) == ((*l2 >> i) & 1)) {
            matches = matches + 1;
        }
    }

    return matches / static_cast<float>(BIT_COUNT);
}

// Average hamming weight
float MatrixHammingWeight(DTYPE *matrix, int rows, int columns) {
    long total = 0;

    for (int i = 0; i < rows * columns; i++) {
        total = total + GetHammingWeight(matrix[i]);
    }

    return static_cast<float>(total) / static_cast<float>(rows * columns);
}

// Average bit alignment
float MatrixBitAlignment(DTYPE *matrix1, DTYPE *matrix2, int rows, int columns) {
    float total = 0;

    for (int i = 0; i < rows * columns; i++) {
        total = total + GetBitAlignment(matrix1[i], matrix2[i]);
    }

    return total / static_cast<float>(rows * columns);
}



// Matrix with RVs from a gaussian distribution
void MatrixGaussian(float *matrix, int rows, int columns, int seed, float *params) {
    float mean = params[0];
    float stddev = params[1];

    DRE.seed(seed);
    std::normal_distribution<float> distribution(mean, stddev);

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = distribution(DRE);
    }
}

// Matrix with RVs from a uniform distribution
void MatrixUniform(float *matrix, int rows, int columns, int seed, float *params) {
    float min = params[0];
    float max = params[1];

    DRE.seed(seed);
    std::uniform_real_distribution<float> distribution(min, max);

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = distribution(DRE);
    }
}

// Matrix with n unique RVs from a gaussian distribution
void MatrixGaussianUnique(float *matrix, int rows, int columns, int seed, float *params) {
    float mean = params[0];
    float stddev = params[1];
    int n = static_cast<int>(params[2]);
    
    DRE.seed(seed);
    std::normal_distribution<float> normal(mean, stddev);
    std::uniform_int_distribution<int> uniform(0, n-1);

    std::vector<float> values;
    for (int i = 0; i < n; i++) {
        values.push_back(normal(DRE));
    }

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = values[uniform(DRE)];
    }
}

// Use for sorting to reverse sort order.
bool reverseOrder(int i, int j) {
    return i > j;
}

// Sort matrix elements row-wise
void MatrixSortRows(float *matrix, int rows, int columns, float percent) {
    if (percent == 0.0) return;

    std::vector<float> values;
    for (int i = 0; i < rows * columns; i++) {
        values.push_back(matrix[i]);
    }
    int m = static_cast<int>(static_cast<float>(rows * columns) * percent);
    std::partial_sort(values.begin(), values.begin()+m, values.end());

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = values[i];
    }
}

// Sort matrix elements column-wise
void MatrixSortColumns(float *matrix, int rows, int columns, float percent) {
    if (percent == 0.0) return;

    std::vector<float> values;
    for (int i = 0; i < rows * columns; i++) {
        values.push_back(matrix[i]);
    }
    int m = static_cast<int>(static_cast<float>(rows * columns) * percent);
    std::partial_sort(values.begin(), values.begin()+m, values.end());

    for (int i = 0; i < values.size(); i++) {
        int r = i % columns;
        int c = i / columns;
        int offset = r * columns + c;
        matrix[offset] = values[i];
    }
}

// Sort matrix elements within rows
void MatrixIntraSortRows(float *matrix, int rows, int columns, float percent) {
    if (percent == 0.0) return;

    int m = static_cast<int>(static_cast<float>(columns) * percent);
    for (int i = 0; i < rows; i++) {
        std::vector<float> values;
        int offset = i * columns;
        for (int j = 0; j < columns; j++) {
            values.push_back(matrix[offset + j]);
        }
        std::partial_sort(values.begin(), values.begin()+m, values.end());
        for (int j = 0; j < columns; j++) {
            matrix[offset + j] = values[j];
        }
    }
}

// Add sparsity to the matrix by setting a percent of values to 0.0 (uniform)
void MatrixAddSparsity(float *matrix, int rows, int columns, int seed, float percent) {
    if (percent == 0.0) return;

    std::default_random_engine DRE;
    DRE.seed(seed);

    std::vector<int> indexes;
    for (int i = 0; i < rows * columns; i++) {
        indexes.push_back(i);
    }
    std::shuffle(indexes.begin(), indexes.end(), DRE);

    int m = static_cast<int>(static_cast<float>(rows * columns) * percent);
    for (int i = 0; i < m; i++) {
        int index = indexes[i];
        matrix[index] = 0.0;
    }
}

// Fill matrix with one initial value
void MatrixInitialValue(float *matrix, int rows, int columns, int seed, float *params) {
    float mean = params[0];
    float stddev = params[1];

    DRE.seed(seed);
    std::normal_distribution<float> normal(mean, stddev);
    float initialValue = normal(DRE);

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = initialValue;
    }
}

// Each entry has n random bits flipped (uniform, no replacement)
void MatrixBits(DTYPE *matrix, int rows, int columns, int seed, float n) {
    int _n = static_cast<int>(n);

    unsigned long one = 1;
    std::vector<size_t> bits;
    for (size_t i = 0; i < BIT_COUNT; i++) {
        bits.push_back(i);
    }

    for (int i = 0; i < rows * columns; i++) {
        std::shuffle(bits.begin(), bits.end(), DRE);
        DTYPE value_copy = matrix[i];
        unsigned long *l = reinterpret_cast<unsigned long *> (&value_copy);
        for (int j = 0; j < _n; j++) {
            *l = *l ^ (one << bits[j]);
        }
        matrix[i] = value_copy;
    }
}

// Each entry has n of the least significant bits randomized (uniform)
void MatrixBitsLeast(DTYPE *matrix, int rows, int columns, int seed, float n) {
    int _n = static_cast<int>(n);

    DRE.seed(seed);
    std::uniform_int_distribution<int> distribution(0, 1);
    unsigned long one = 1;
    
    for (int i = 0; i < rows * columns; i++) {
        DTYPE value_copy = matrix[i];
        unsigned long *l = reinterpret_cast<unsigned long *>(&value_copy);
        for (int j = 0; j < _n; j++) {
            if (distribution(DRE) == 0) {
                *l = *l & ~(one << j);
            } else {
                *l = *l | (one << j);
            }
        }
        matrix[i] = value_copy;
    }
}

// Each entry has n of the most significant bits randomized (uniform)
void MatrixBitsMost(DTYPE *matrix, int rows, int columns, int seed, float n) {
    int _n =  static_cast<int>(n);

    DRE.seed(seed);
    std::uniform_int_distribution<int> distribution(0, 1);
    unsigned long one = 1;

    for (int i = 0; i < rows * columns; i++) {
        DTYPE value_copy = matrix[i];
        unsigned long *l = reinterpret_cast<unsigned long *>(&value_copy);
        for (int j = static_cast<int>(BIT_COUNT) - 1; j > static_cast<int>(BIT_COUNT) - _n - 1; j--) {
            if (distribution(DRE) == 0) {
                *l = *l & ~(one << j);
            } else {
                *l = *l | (one << j);
            }
        }
        matrix[i] = value_copy;
    }
}

// Set least significant bits to zero
void MatrixSetBitsLeastZero(DTYPE *matrix, int rows, int columns, float zeros) {
    if (zeros == 0.0) return;

    int n = static_cast<int>(zeros);
    unsigned long one = 1;
    
    for (int i = 0; i < rows * columns; i++) {
        DTYPE value_copy = matrix[i];
        unsigned long *l = reinterpret_cast<unsigned long *>(&value_copy);
        for (int j = 0; j < n; j++) {
            *l = *l & ~(one << j);
        }
        matrix[i] = value_copy;
    }
}

// Set most significant bits to zero
void MatrixSetBitsMostZero(DTYPE *matrix, int rows, int columns, float zeros) {
    if (zeros == 0.0) return;

    int n = static_cast<int>(zeros);
    unsigned long one = 1;

    for (int i = 0; i < rows * columns; i++) {
        DTYPE value_copy = matrix[i];
        unsigned long *l = reinterpret_cast<unsigned long *>(&value_copy);
        for (int j = static_cast<int>(BIT_COUNT) - 1; j > static_cast<int>(BIT_COUNT) - n - 1; j--) {
            *l = *l & ~(one << j);
        }
        matrix[i] = value_copy;
    }
}

// Transpose the matrix
void MatrixTranspose(float *matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = i; j < columns; j++) {
            float temp = matrix[i * columns + j];
            matrix[i * columns + j] = matrix[j * columns + i];
            matrix[j * columns + i] = temp;
        }
    }
}
