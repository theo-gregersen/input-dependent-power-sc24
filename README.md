# About

The code and data in this repository were used for the paper "Input-Dependent Power Usage in GPUs" published in SC'24 Sustainable Supercomputing workshop.

# Setup

- Install [Miniconda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) and make a python environment (e.g., `conda create -n gemm python`).
- Activate the conda environment (e.g., `conda activate gemm`) and install necessary packages (e.g., `pip install cmake` and `pip install numpy`).
- Install [Cuda Toolkit 12.2+](https://developer.nvidia.com/cuda-toolkit) and set environment variable `CUDACXX` to point to the `cuda-x.y/bin/nvcc` directory (or update path).
- Install [Nvidia DCGM](https://developer.nvidia.com/dcgm) and enable the [DCGM systemd service](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html#post-install).
- These experiments stem from the Cutlass examples. Clone the [Cutlass](https://github.com/NVIDIA/cutlass) repo.
- Move or copy `matrices.h` to the `cutlass/include/cutlass/gemm/` directory.
- Move or copy `basic_gemm_expt.cu` and `other_expt.cu` to the `cutlass/examples/00_basic_gemm/` directory.
- Add the following to the `cutlass/examples/00_basic_gemm/CMakeLists.txt` file:

    ```
    cutlass_example_add_executable(
        00_basic_gemm_expt
        basic_gemm_expt.cu
    )

    cutlass_example_add_executable(
        other_expt
        other_expt.cu
    )
    ```

- You might need to comment out the `swap` method in `cutlass/include/cutlass/fast_math.h` due to overrides.

# Build

Add a build folder in the main `cutlass/` directory.

`mkdir build & cd build`

Standard compile for A100. Change the arch code to compile for other platforms, or see Cutlass documentation for more options. The arch code for A100 is 80, H100 is 90a, and RTX 6000 is 75 ([this](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) is a useful website on NVIDIA arch codes).

`cmake .. -DCUTLASS_NVCC_ARCHS=80`

If you want to use tensor cores, compile with:

`cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_F16C_ENABLED=ON -DCUTLASS_LIBRARY_KERNELS=tensorop`

Navigate to the build and make.

`cd examples/00_basic_gemm && make`

# Run

To change the datatype used in the experiments, update the `DTYPE` field at the top of the `matrices.h` file. Note that all RVs are initially generated as floats and then converted to `DTYPE` with the [Cutlass NumericConverter](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h). The default float rounding style is round to nearest.

The experiment script is `00_basic_gemm_expt`, located in `cutlass/build/examples/00_basic_gemm`. It has the following usage:

`./00_basic_gemm_expt <M> <N> <K> <alpha> <beta> <iterations> <seed> <pattern> <output path> <pattern parameters>`

Example parameters for each pattern can be seen in `paper_expts.py`. The paper experiments are also in `paper_expts.py`.

To run experiment 13 (no transpose on B), set the transpose parameter of the B matrix to be `false` in the `TestCutlassGemm` function of `basic_gemm_expt.cu`.

To run experiments with tensor cores enabled, uncomment the `using MMAOp` and `using SmArch` statements in the `CutlassSgemmNN` function of `basic_gemm_expt.cu`. Add the following parameters to the subsequent `using CutlassGemm` statement: `DTYPE, MMAOp, SmArch`.
