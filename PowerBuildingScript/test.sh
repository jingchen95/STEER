#!/usr/bin/env bash
#SBATCH -A SNIC2019-3-293 -p hebbe
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 0-00:03:00
#SBATCH --error=job.%J.err 
#SBATCH --output=test.txt


export XITAO_LAYOUT_PATH=../ptt_layout_files/s0_c2
echo "Socket 0 - 2 cores: "
../benchmarks/syntheticDAGs/synbench 64 8192 0 2 0 100 0 1