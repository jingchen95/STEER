#!/bin/bash
export XITAO_LAYOUT_PATH=./ptt_layout_tx2

# 2022 - Feb 2nd - EDP optimization per task
# parallelism="8"
# for dop in $parallelism
# do
#     for((k=0;k<10;k++))
#     do
#         echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 1
#         ./benchmarks/syntheticDAGs/synbench 1 0 2048 0 1 0 50000 0 $dop > cp_2048_50k_EDP_${k}.txt
#         sleep 5
#     done
#     # for((k=0;k<2;k++))
#     # do
#     #     echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#     #     echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#     #     sleep 1
#     #     ./benchmarks/syntheticDAGs/synbench 1 0 0 1024 1 0 0 50000 $dop > st_1024_50k_EDP_${k}.txt
#     #     sleep 5
#     # done
#     for((k=0;k<10;k++))
#     do
#         echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 1
#         ./benchmarks/syntheticDAGs/synbench 1 256 0 0 1 10000 0 0 $dop > mm_256_10k_EDP_${k}.txt 
#         sleep 5
#     done
# done

# --- 2D Heat ---
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "2D Heat - SEER Begin the $k execution! "
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 2
#         ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/big.dat > heat_10k_2048_EDP_${k}.txt
#         sleep 5
#         echo "2D Heat - SEER End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# --- Sparse LU ---
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Sparse LU - SEER Begin the $k execution! "
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 2
#         ./benchmarks/sparselu/sparselu 1 64 512 > slu_EDP_64_${k}.txt
#         sleep 10
#         echo "Sparse LU - SEER End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"
# done

for((k=0;k<10;k++))
do
        echo "/*---------------------------------------------------------------*/"
        echo "Sparse LU - SEER Begin the $k execution! "
        echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        sleep 2
        ./benchmarks/sparselu/sparselu 1 32 512 > slu_EDP_32_${k}.txt
        sleep 10
        echo "Sparse LU - SEER End the $k execution! "
        echo "/*---------------------------------------------------------------*/"
done

# --- 2D Heat ---
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "SEER Begin the $k execution! "
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 2
#         ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/big.dat > ./debug_results/heat_10k_2048_${k}.txt
#         sleep 5
#         echo "SEER End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# --- Dot Product ---
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "SEER Begin the $k execution! "
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 2
        # width = 1, block length = 640000, 100 blocks = 100 tasks, 100 iterations ==> 10000 tasks in total 
#         ./benchmarks/dotproduct/dotprod 1 100 64000000 1 320000 > ./debug_results/dotprod_${k}.txt
#         sleep 2
#         echo "SEER End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"
# done

# --- Fibnacci ---
# for((k=0;k<10;k++))
# do
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 2
#         # 57236 tasks with term 55 grain_size 34
#         ./benchmarks/fibonacci/fibonacci 1 55 34 > ./debug_results/fibonacci.txt
#         sleep 2
# done

# --- Sparse LU ---
# for((k=0;k<7;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "SEER Begin the $k execution! "
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 2
#         ./benchmarks/sparselu/sparselu 1 64 512 > ./debug_results/sparselu_${k}.txt
#         sleep 10
#         echo "SEER End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"
# done

# --- Synthetic DAGs ---
# parallelism="4 6 8"
# for dop in $parallelism
# do
#     for((k=0;k<10;k++))
#     do
        # Single Kernel Case
        # echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        # echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        # sleep 1 
        # ./benchmarks/syntheticDAGs/synbench 1 64 0 0 1 50000 0 0 $dop > ./debug_results/mm_64_50k_p${dop}_$k.txt 
        # sleep 5
        # echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        # echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        # sleep 1
        # ./benchmarks/syntheticDAGs/synbench 1 256 0 0 1 10000 0 0 $dop > ./debug_results/mm_256_10k_p${dop}_${k}.txt 
        # sleep 5
        # echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        # echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        # sleep 1
        # ./benchmarks/syntheticDAGs/synbench 1 0 1024 0 1 0 50000 0 $dop > ./debug_results/cp_1024_50000_p${dop}_${k}.txt
        # ./benchmarks/syntheticDAGs/synbench 1 0 2048 0 1 0 50000 0 $dop > ./debug_results/cp_2048_50000_p${dop}_${k}.txt
        # sleep 5
        # echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        # echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        # sleep 1
        # ./benchmarks/syntheticDAGs/synbench 1 0 0 512 1 0 0 50000 $dop > ./debug_results/st_512_50000_p${dop}_${k}.txt
        # ./benchmarks/syntheticDAGs/synbench 1 0 0 1024 1 0 0 50000 $dop > ./debug_results/st_1024_50000_p${dop}_${k}.txt
        # sleep 5

        # Multiple Kernels Case (does not work!)
        # echo 2035200  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        # echo 2035200  > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        # sleep 1
        # ./benchmarks/syntheticDAGs/synbench 1 256 2048 0 1 500 500 0 $dop > ./debug_results/mm+cp_$k.txt
        # sleep 5
#     done
# done

