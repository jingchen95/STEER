#!/bin/bash
export XITAO_LAYOUT_PATH=./ptt_layout_tx2

for((k=0;k<10;k++))
do
  DENVER=0 A57=0 ./benchmarks/syntheticDAGs/synbench 0 64 0 0 1 50000 0 0 4  
done
