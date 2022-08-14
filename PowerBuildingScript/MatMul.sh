#!/usr/bin/env bash
#SBATCH -A SNIC2019-3-293 -p hebbe
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 0-00:60:00
#SBATCH --error=job.%J.err 
#SBATCH --output=EAS_MatMul.txt

../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../../EAS_XITAO_c1/ptt_layout_files/s0_c1
echo "Socket 0 - 1 core: "
../../EAS_XITAO_c1/benchmarks/syntheticDAGs/synbench 512 0 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../ptt_layout_files/s0_c2
echo "Socket 0 - 2 cores: "
../benchmarks/syntheticDAGs/synbench 3 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../ptt_layout_files/s0_c5
echo "Socket 0 - 5 cores: "
../benchmarks/syntheticDAGs/synbench 3 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../ptt_layout_files/s0_c10
echo "Socket 0 - 10 cores: "
../benchmarks/syntheticDAGs/synbench 3 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../../EAS_XITAO_c1/ptt_layout_files/s1_c1
echo "Socket 1 - 1 core: "
../../EAS_XITAO_c1/benchmarks/syntheticDAGs/synbench 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../ptt_layout_files/s1_c2
echo "Socket 1 - 2 cores: "
../benchmarks/syntheticDAGs/synbench 3 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../ptt_layout_files/s1_c5
echo "Socket 1 - 5 cores: "
../benchmarks/syntheticDAGs/synbench 3 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot

sleep 60

export XITAO_LAYOUT_PATH=../ptt_layout_files/s1_c10
echo "Socket 1 - 10 cores: "
../benchmarks/syntheticDAGs/synbench 3 512 16384 0 1 80 0 0 12 &
../../uarch-configure/rapl-read/rapl-plot