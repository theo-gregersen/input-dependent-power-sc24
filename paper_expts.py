import os
import numpy as np
import time

DATATYPE = 'fp16'
MAX_BIT_FLIPS = 8 if DATATYPE == 'int8' else 16
OUTPUT_DIR = f'expt_results/{DATATYPE}'
SCRIPT_PATH = 'cutlass/build/examples/00_basic_gemm/00_basic_gemm_expt'

STD_MEAN = 0
STD_STDDEV = 2**10 # 2**10 for FP, 2**5 for int8
MIN_MEAN_EXP = 0 # 0 for FP, 0 for int8
MAX_MEAN_EXP = 14 # 14 for FP, 6 for int8
MIN_STDDEV_EXP = -5 # -5 for FP, 0 for int8
MAX_STDDEV_EXP = 10 # 10 for FP, 6 for int8

N = 2048
iterations = 10000 # 10000 for non tensor core, 20000 for tensor core
seeds = np.arange(1, 11) # C DRE gives same output for seeds 0 and 1
alpha = 1
beta = 0
base_cmd = f'{SCRIPT_PATH} {N} {N} {N} {alpha} {beta} {iterations}'

def expt1_gaussian_stddev(output = f'{OUTPUT_DIR}/1-gaussian-stddev', min_exp = MIN_STDDEV_EXP, max_exp = MAX_STDDEV_EXP):
    for seed in seeds:
        mean = 0
        for stddev in [np.round(2.0**n, 4) for n in np.arange(min_exp, max_exp+1)]:
            file = f'{output}/{seed}_{stddev}.tsv'
            os.system(f'{base_cmd} {seed} gaussian {file} {mean} {stddev} >> expt_results/bit_metrics/{DATATYPE}/1-gaussian-stddev.csv')

def expt2_gaussian_mean(output = f'{OUTPUT_DIR}/2-gaussian-mean', min_exp = MIN_MEAN_EXP, max_exp = MAX_MEAN_EXP):
    for seed in seeds:
        stddev = 2**0 # 2**0 for FP, 2**3 for int8
        for mean in [2**n for n in np.arange(min_exp, max_exp+1)]:
            file = f'{output}/{seed}_{mean}.tsv'
            os.system(f'{base_cmd} {seed} gaussian {file} {mean} {stddev} >> expt_results/bit_metrics/{DATATYPE}/2-gaussian-mean.csv')

def expt3_n_values(output = f'{OUTPUT_DIR}/3-gaussian-unique', mean = STD_MEAN, stddev = STD_STDDEV):
    for seed in seeds:
        for n in [2**n for n in np.arange(0, 10)]:
            file = f'{output}/{seed}_{n}.tsv'
            os.system(f'{base_cmd} {seed} unique {file} {mean} {stddev} {n} >> expt_results/bit_metrics/{DATATYPE}/3-gaussian-unique.csv')

def bits_expt(pattern, output, mean = STD_MEAN, stddev = STD_STDDEV):
    for seed in seeds:
        for n in np.arange(0, MAX_BIT_FLIPS+1):
            file = f'{output}/{seed}_{n}.tsv'
            os.system(f'{base_cmd} {seed} {pattern} {file} {mean} {stddev} {n} >> expt_results/bit_metrics/{DATATYPE}/{pattern}.csv')

def expt7_sort_mean(output = f'{OUTPUT_DIR}/7-sort-mean', min_exp = MIN_MEAN_EXP, max_exp = MAX_MEAN_EXP):
    for seed in seeds:
        stddev = 2**0 # 2**0 for FP, 2**3 for int8
        for mean in [2**n for n in np.arange(min_exp, max_exp+1)]:
            file = f'{output}/{seed}_{mean}.tsv'
            os.system(f'{base_cmd} {seed} sort_rows {file} {mean} {stddev} {1.0} >> expt_results/bit_metrics/{DATATYPE}/7-sort-mean.csv') 

def baseline(output = f'{OUTPUT_DIR}/baseline', mean = STD_MEAN, stddev = STD_STDDEV):
    for seed in seeds:
        file = f'{output}/{seed}.tsv'
        os.system(f'{base_cmd} {seed} gaussian {file} {mean} {stddev} >> expt_results/bit_metrics/{DATATYPE}/baseline.csv')

def baseline_percent_expt(pattern, output, mean = STD_MEAN, stddev = STD_STDDEV):
    for seed in seeds:
        for p in np.round(np.arange(0, 1.1, 0.1), 1):
            file = f'{output}/{seed}_{p}.tsv'
            os.system(f'{base_cmd} {seed} {pattern} {file} {mean} {stddev} {p} >> expt_results/bit_metrics/{DATATYPE}/13-sort-no-transpose.csv')

# 1) RVs from gaussian, vary standard deviation
expt1_gaussian_stddev()

# 2) RVs from gaussian, vary mean
expt2_gaussian_mean()

# 3) n unique values from gaussian, vary n
expt3_n_values()

# 4) n random bits flipped, vary n
bits_expt('bits', f'{OUTPUT_DIR}/4-bits')

# 5) n least significant bits randomized, vary n
bits_expt('bits_least', f'{OUTPUT_DIR}/5-bits-least')

# 6) n most significant bits randomized, vary n
bits_expt('bits_most', f'{OUTPUT_DIR}/6-bits-most')

# 7) RVs from gaussian, vary mean, fully sort
expt7_sort_mean()

# Baseline for following experiments
baseline()

# 8) Partial sort p percent of the matrix row-wise, vary p
baseline_percent_expt('sort_rows', f'{OUTPUT_DIR}/8-sort-rows')

# 9) Partial sort p percent of the matrix column-wise, vary p
baseline_percent_expt('sort_columns', f'{OUTPUT_DIR}/9-sort-columns')
            
# 10) Partial sort p percent of the matrix per row, vary p
baseline_percent_expt('sort_rows_intra', f'{OUTPUT_DIR}/10-sort-intra')

# 11) Make p percent of the matrix sparse, vary p
baseline_percent_expt('sparsity', f'{OUTPUT_DIR}/11-sparsity')

# 12) Fully sort the matrix, make p percent of the matrix sparse, vary p
baseline_percent_expt('sort+sparsity', f'{OUTPUT_DIR}/12-sparsity-sort')

# 13) Partial sort p percent of the matrix row-wise, vary p, don't transpose B
baseline_percent_expt('sort_rows', f'{OUTPUT_DIR}/13-sort-no-transpose')

# 14) n least significant bits set to 0, vary n
bits_expt('bits_least_zeros', f'{OUTPUT_DIR}/14-bits-least-zeros')

# 15) n most significant bits set to 0, vary n
bits_expt('bits_most_zeros', f'{OUTPUT_DIR}/15-bits-most-zeros')

# Other experiments

N = 2**16
seconds = 300
OUTPUT_DIR = 'expt_results/other_expt'

# 16) memory - ones: fill matrix with ones, record power
os.system(f'cutlass/build/examples/00_basic_gemm/other_expt {N} {N} ones {OUTPUT_DIR}/ones.tsv 0 {seconds}')

# 17) memory - zeros: fill matrix with zeros: record power
os.system(f'cutlass/build/examples/00_basic_gemm/other_expt {N} {N} zeros {OUTPUT_DIR}/zeros.tsv 0 {seconds}')

iterations = 10000
N = 2**14

# 18) bit flips - kernel: fill matrix with zeros, flip the bits n times
# warmup
os.system(f'cutlass/build/examples/00_basic_gemm/other_expt {N} {N} bit-flips {OUTPUT_DIR}/bit_flips/warmup.tsv {1000000} {0}')
for i in np.arange(33):
    os.system(f'cutlass/build/examples/00_basic_gemm/other_expt {N} {N} bit-flips {OUTPUT_DIR}/bit_flips/{i}.tsv {iterations} {i}')

# Measure idle power
os.system(f'dcgmi dmon -e 100,101,112,140,150,155,156,190,191,203,204,206,207,210,211,240,241,242,243,244,245,246,247,252,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -d 100 > {OUTPUT_DIR}/idle.tsv &')
time.sleep(300)
os.system('pkill -f dcgmi')
