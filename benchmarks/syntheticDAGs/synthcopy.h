#ifndef SYNTH_COPY
#define SYNTH_COPY

#include "tao.h"
#include "dtypes.h"
#include <chrono>
#include <iostream>
#include <atomic>
#include <cmath>
#include <stdio.h>
#define PSLACK 8  

// Matrix multiplication, tao groupation on written value
class Synth_MatCopy : public AssemblyTask 
{
public: 
  static float time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static float power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static uint64_t cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static float mb_table[NUMSOCKETS][XITAO_MAXTHREADS]; /*mb - memory-boundness */
  static bool time_table_state[NUMSOCKETS+1];
  static bool best_config_state;
  static bool enable_freq_change;
  static int best_freqindex;
  static int best_cluster;
  static int best_width;
  static std::atomic<int> PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static std::atomic<int> PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];

  Synth_MatCopy(uint32_t _size, int _width, real_t *_A, real_t *_B): AssemblyTask(_width), A(_A), B(_B) {   
    dim_size = _size;
    block_index = 0;
    // block_size = dim_size / (_width * PSLACK);
    // if(block_size == 0) block_size = 1;
    // uint32_t elem_count = dim_size * dim_size;
    // block_count = dim_size / block_size;
  }

  void cleanup(){ }

  // this assembly can work totally asynchronously
  void execute(int threadid) {
    // while(true) {
    //   int row_block_id = block_index++;
    //   if(row_block_id > block_count) return;
    //   int row_block_start =  row_block_id      * block_size;
    //   int row_block_end   = (row_block_id + 1) * block_size;
    //   int end = (dim_size < row_block_end) ? dim_size : row_block_end; 
    //   for (int i = row_block_start; i < end; ++i) { 
    //      std::copy(A + (i * dim_size), A + (i * dim_size) + dim_size, B + i * dim_size);
    //   }
    // }
    
    // Add by Jing
    block_size = dim_size / (width * PSLACK);
    if(block_size == 0) block_size = 1;
    block_count = dim_size / block_size;
// #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "Task " << taskid << ", width: " << width << ". total number of blocks: " << block_count << ". Thread " << threadid << " will do block " << (threadid-leader) * (block_count/width) \
//     << " to block " << ((threadid-leader)+1) * (block_count/width) << std::endl;
//     LOCK_RELEASE(output_lck);
// #endif
    for (int i = (threadid-leader) * (block_count/width)* block_size; i < dim_size && i < ((threadid-leader)+1) * (block_count/width)* block_size; ++i) {
      std::copy(A + (i * dim_size), A + (i * dim_size) + dim_size, B + i * dim_size);
    }
  }

  void increment_PTT_UpdateFinish(int freq_index, int clusterid, int index) {
    PTT_UpdateFinish[freq_index][clusterid][index]++;
  }
  float get_PTT_UpdateFinish(int freq_index, int clusterid,int index){
    float finish = 0;
    finish = PTT_UpdateFinish[freq_index][clusterid][index];
    return finish;
  }
  void increment_PTT_UpdateFlag(int freq_index, int clusterid, int index) {
    PTT_UpdateFlag[freq_index][clusterid][index]++;
  }
  float get_PTT_UpdateFlag(int freq_index, int clusterid,int index){
    float finish = 0;
    finish = PTT_UpdateFlag[freq_index][clusterid][index];
    return finish;
  }
  void set_timetable(int freq_index, int clusterid, float ticks, int index) {
    time_table[freq_index][clusterid][index] = ticks;
  }
  float get_timetable(int freq_index, int clusterid, int index) { 
    float time=0;
    time = time_table[freq_index][clusterid][index];
    return time;
  }
  void set_powertable(int freq_index, int clusterid, float power_value, int index) {
    power_table[freq_index][clusterid][index] = power_value;
  }
  float get_powertable(int freq_index, int clusterid, int index) { 
    float power_value = 0;
    power_value = power_table[freq_index][clusterid][index];
    return power_value;
  }
  void set_cycletable(int freq_index, int clusterid, uint64_t cycles, int index) {
    cycle_table[freq_index][clusterid][index] = cycles;
  }
  uint64_t get_cycletable(int freq_index, int clusterid, int index) { 
    uint64_t cycles = 0;
    cycles = cycle_table[freq_index][clusterid][index];
    return cycles;
  }
  void set_mbtable(int clusterid, float mem_b, int index) {
    mb_table[clusterid][index] = mem_b;
  }
  float get_mbtable(int clusterid, int index) { 
    float mem_b = 0;
    mem_b = mb_table[clusterid][index];
    return mem_b;
  }
  bool get_timetable_state(int cluster_index){
    bool state = time_table_state[cluster_index];
    return state;
  }
  void set_timetable_state(int cluster_index, bool new_state){
    time_table_state[cluster_index] = new_state;
  }
  /* Find out the best config for this kernel task */
  bool get_bestconfig_state(){
    bool state = best_config_state;
    return state;
  }
  void set_bestconfig_state(bool new_state){
    best_config_state = new_state;
  }
  /* Enable frequency change or not (fine-grained or coarse-grained) */
  bool get_enable_freq_change(){
    bool state = enable_freq_change;
    return state;
  }
  void set_enable_freq_change(bool new_state){
    enable_freq_change = new_state;
  }
  void set_best_freq(int freq_index){
    best_freqindex = freq_index;
  }
  void set_best_cluster(int clusterid){
    best_cluster = clusterid;
  }
  void set_best_numcores(int width){
    best_width = width;
  }
  int get_best_freq(){
    int freq_indx = best_freqindex;
    return freq_indx;
  }
  int get_best_cluster(){
    int clu_id = best_cluster;
    return clu_id;
  }
  int get_best_numcores(){
    int wid = best_width;
    return wid;
  }
private:
  std::atomic<int> block_index; 
  int block_size; 
  int dim_size;
  int block_count;
  real_t *A, *B;
};
#endif
