/* assembly_sched.cxx -- integrated work stealing with assembly scheduling */
#include "tao.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <time.h>
#include <sstream>
#include <cstring>
#include <unistd.h> 
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <numeric>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include "xitao_workspace.h"
using namespace xitao;

const int EAS_PTT_TRAIN = 2;

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[];
};
// std::ofstream pmc;
std::ofstream Denver("/sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
std::ofstream ARM("/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");

#if defined(Haswell)
int num_sockets;
#endif

int a57_freq;
int denver_freq;

#if (defined DynaDVFS)
// int freq_dec;
// int freq_inc;
// 0 denotes highest frequency, 1 denotes lowest. 
// e.g. in 00, first 0 is denver, second 0 is a57. env= 0*2+0 = 0
int env;
// int dynamic_time_change;
int current_freq;
#endif

int freq_index;
long cur_freq[NUMSOCKETS] = {2035200, 2035200}; /*starting frequency is 2.04GHz for both clusters */
int cur_freq_index[NUMSOCKETS] = {0,0};
long avail_freq[NUM_AVAIL_FREQ] = {2035200, 1881600, 1728000, 1574400, 1420800, 1267200, 1113600, 960000, 806400, 652800, 499200, 345600};
int num_width[NUMSOCKETS] = {2, 3};
int ptt_freq_index[NUMSOCKETS] = {0};
// int PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS] = {0};
int start_coreid[NUMSOCKETS] = {0, 2};
int end_coreid[NUMSOCKETS] = {2, XITAO_MAXTHREADS};
int PTT_finish_state[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS] = {0}; // First: 2.04 or 1.11; Second: two clusters; third: the number of kernels (assume < XITAO_MAXTHREADS)
int global_training_state[XITAO_MAXTHREADS] = {0}; // array size: the number of kernels (assume < XITAO_MAXTHREADS)
bool global_training = false;

int status[XITAO_MAXTHREADS];
int status_working[XITAO_MAXTHREADS];
int Sched, num_kernels;
int maySteal_DtoA, maySteal_AtoD;
std::atomic<int> DtoA(0);

// define the topology
int gotao_sys_topo[5] = TOPOLOGY;

#ifdef NUMTASKS
int NUM_WIDTH_TASK[XITAO_MAXTHREADS] = {0};
#endif

#ifdef DVFS
float compute_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS] = {0.0};
float memory_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS] = {0.0};
float cache_intensive_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS] = {0.0};
#elif (defined DynaDVFS) // Currently only consider 4 combinations: max&max, max&min, min&max, min&min
float compute_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float memory_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float cache_intensive_power[4][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
#elif (defined ERASE)
float compute_bound_power[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float memory_bound_power[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float cache_intensive_power[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
#else
float runtime_power[10][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0}; // TX2 Power Profiles: 10 groups by memory-boundness level
float idle_power[NUM_AVAIL_FREQ][NUMSOCKETS] = {0.0};
#endif

struct timespec tim, tim2;
cpu_set_t affinity_setup;
int TABLEWIDTH;
int worker_loop(int);

#ifdef PowerProfiling
std::ofstream out("KernelTaskTime.txt");
#endif

#ifdef NUMTASKS_MIX
//std::vector<int> num_task(XITAO_MAXTHREADS * XITAO_MAXTHREADS, 0);
int num_task[XITAO_MAXTHREADS][XITAO_MAXTHREADS * XITAO_MAXTHREADS] = {0}; /*First parameter: assume an application has XITAO_MAXTHREADS different kernels at most*/
#endif

int PTT_flag[XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::chrono::time_point<std::chrono::system_clock> t3;
std::mutex mtx;
std::condition_variable cv;
bool finish = false;

// std::vector<thread_info> thread_info_vector(XITAO_MAXTHREADS);
//! Allocates/deallocates the XiTAO's runtime resources. The size of the vector is equal to the number of available CPU cores. 
/*!
  \param affinity_control Set the usage per each cpu entry in the cpu_set_t
 */
int set_xitao_mask(cpu_set_t& user_affinity_setup) {
  if(!gotao_initialized) {
    resources_runtime_conrolled = true;                                    // make this true, to refrain from using XITAO_MAXTHREADS anywhere
    int cpu_count = CPU_COUNT(&user_affinity_setup);
    runtime_resource_mapper.resize(cpu_count);
    int j = 0;
    for(int i = 0; i < XITAO_MAXTHREADS; ++i) {
      if(CPU_ISSET(i, &user_affinity_setup)) {
        runtime_resource_mapper[j++] = i;
      }
    }
    if(cpu_count < gotao_nthreads) std::cout << "Warning: only " << cpu_count << " physical cores available, whereas " << gotao_nthreads << " are requested!" << std::endl;      
  } else {
    std::cout << "Warning: unable to set XiTAO affinity. Runtime is already initialized. This call will be ignored" << std::endl;      
  }  
}

void gotao_wait() {
//  gotao_master_waiting = true;
//  master_thread_waiting.notify_one();
//  std::unique_lock<std::mutex> lk(pending_tasks_mutex);
//  while(gotao_pending_tasks) pending_tasks_cond.wait(lk);
////  gotao_master_waiting = false;
//  master_thread_waiting.notify_all();
  while(PolyTask::pending_tasks > 0);
}
//! Initialize the XiTAO Runtime
/*!
  \param nthr is the number of XiTAO threads 
  \param thrb is the logical thread id offset from the physical core mapping
  \param nhwc is the number of hardware contexts
*/ 
int gotao_init_hw( int nthr, int thrb, int nhwc)
{
  gotao_initialized = true;
 	if(nthr>=0) gotao_nthreads = nthr;
  else {
    if(getenv("GOTAO_NTHREADS")) gotao_nthreads = atoi(getenv("GOTAO_NTHREADS"));
    else gotao_nthreads = XITAO_MAXTHREADS;
  }
  if(gotao_nthreads > XITAO_MAXTHREADS) {
    std::cout << "Fatal error: gotao_nthreads is greater than XITAO_MAXTHREADS of " << XITAO_MAXTHREADS << ". Make sure XITAO_MAXTHREADS environment variable is set properly" << std::endl;
    exit(0);
  }

#if defined(TX2)
	if(nhwc>=0){
    a57_freq = nhwc;
  }
  else{
    if(getenv("A57")){
      a57_freq = atoi(getenv("A57"));
    }
    else{
      a57_freq = A57;
    }
  } 

	if(nhwc>=0){
    denver_freq = nhwc;
  }
  else{
    if(getenv("DENVER")){
      denver_freq = atoi(getenv("DENVER"));
    }
    else{
      denver_freq = DENVER;
    }
  }
#endif
  
  /* 2021 Oct 02: Read Power Profile File, including idle and dynamic power */
  std::ifstream infile, infile1;
  infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2/TX2_idle.txt"); /* Step 1: Read idle power */
  if(infile1.fail()){
    std::cout << "Failed to open power profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token1;
  while(std::getline(infile1, token1)) {
    std::istringstream line(token1);
    int ii = 0; // Column index of power files
    int freq = 0; // Frequency Index ranging from 0 to 11
    float idlep = 0;
    while(line >> token1) {
      if(ii == 0){
        freq = stoi(token1); // first column is frequency index
      }else{
        idlep = stof(token1);
        idle_power[freq][ii-1] = idlep; 
      }
      ii++;
    }
  }
  infile1.close();

  for(int mb_bound = 0; mb_bound < 10; mb_bound++){ /* Step 2: Read runtime power */
    char address[100] = {'\0'};
    sprintf(address, "/home/nvidia/Desktop/EAS/PowerProfile/TX2/TX2_%d_%d.txt", mb_bound, mb_bound+1);
    // std::cout << "Trying to open file " << address << "...\n";
    // FILE * files = fopen(address, "r");
    // if (files == NULL) {
    //   perror("Couldn't open file");
    // } else {
    //   printf("Successfully opened file %s\n", address);
    //   fclose(files);
    // }
    infile.open(address);
    if(infile.fail()){
      std::cout << "Failed to open power profile file!" << std::endl;
      std::cin.get();
      return 0;
    }
    std::string token;
    while(std::getline(infile, token)) {
      std::istringstream line(token);
      int ii = 0; // Column index of power files
      int avail_width = 0;
      int freq = 0; // Frequency Index ranging from 0 to 11
      float runtimep = 0; // Read runtime power
      while(line >> token) {
        if(ii == 0){
          freq = stoi(token); // first column is frequency index
        }else{
          if(ii == 1){
            avail_width = stoi(token); // Second column is the available widths
          }
          else{
            runtimep = stof(token); // Other two columns are Denver and A57 runtime power respectively
            runtime_power[mb_bound][freq][ii-2][avail_width-1] = runtimep; 
          } 
        }
        ii++;
      }
    }
    infile.close();
  }
/*
#ifdef DEBUG
  std::cout << "************* TX2 Power Profiles *************\n";
  // Output the idle power
  for(int kk = 0; kk < NUM_AVAIL_FREQ; kk++){
    std::cout << "Freq: " << avail_freq[kk] << "\n";
    for(int ii = 0; ii < NUMSOCKETS; ii++){
      std::cout << idle_power[kk][ii] << "\t";
    }
    std::cout << std::endl;
  }
  // Output the runtime power
  for(int ww = 0; ww < 10; ww++){
    std::cout << "Memory-boundness (" << ww << ", " << ww+1 << "): \n";
    for(int kk = 0; kk < NUM_AVAIL_FREQ; kk++){
      std::cout << "Freq: " << avail_freq[kk] << "\n";
      for(int ii = 0; ii < NUMSOCKETS; ii++){
        std::cout << "Cluster " << ii << ": ";
        for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
          if(runtime_power[ww][kk][ii][jj] == 0){
            std::cout << "---\t";
          }else{
            std::cout << runtime_power[ww][kk][ii][jj] << "\t";
          }
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
#endif
*/
#if 0  
#ifdef Haswell  
  infile.open("/cephyr/users/chjing/Hebbe/EAS_XITAO/PowerProfile/Hebbe_MatMul.txt");
#endif

#if (defined DVFS) && (defined TX2)
// Needs to find out the task type ??????
  infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_DVFS_MM");
  if(infile.fail()){
    std::cout << "Failed to open power profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token;
  while(std::getline(infile, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    int freq = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        freq = stoi(token);
      }else{
        if(ii == 1){
          pwidth = stoi(token);
        }
        else{
          get_power = stof(token);
          compute_bound_power[freq][ii-2][pwidth] = get_power; 
        } 
      }
      ii++;
      //std::cout << "Token :" << token << std::endl;
    }
    // if(infile.unget().get() == '\n') {
    //   std::cout << "newline found" << std::endl;
    // }
  }

  // Output the power reading
  // for(int kk = 0; kk < FREQLEVELS; kk++){
  //   for(int ii = 0; ii < NUMSOCKETS; ii++){
  //     for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
  //       std::cout << compute_bound_power[kk][ii][jj] << "\t";
  //     }
  //     std::cout << "\n";
  //   }
  // }

  infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_DVFS_CP");
  if(infile1.fail()){
    std::cout << "Failed to open power profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  //std::string token;
  while(std::getline(infile1, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    int freq = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        freq = stoi(token);
      }else{
        if(ii == 1){
          pwidth = stoi(token);
        }
        else{
          get_power = stof(token);
          memory_bound_power[freq][ii-2][pwidth] = get_power; 
        } 
      }
      ii++;
      //std::cout << "Token :" << token << std::endl;
    }
    // if(infile.unget().get() == '\n') {
    //   std::cout << "newline found" << std::endl;
    // }
  }

  // Output the power reading
  // for(int kk = 0; kk < FREQLEVELS; kk++){
  //   for(int ii = 0; ii < NUMSOCKETS; ii++){
  //     for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
  //       std::cout << memory_bound_power[kk][ii][jj] << "\t";
  //     }
  //     std::cout << "\n";
  //   }
  // }

#elif (defined CATA)
  // Denver Frequency: 2035200, A57 Frequency: 1113600
  infile.open("/home/nvidia/Desktop/EAS/PowerProfile/COMP_CATA.txt");
  std::string token;
  while(std::getline(infile, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        compute_bound_power[ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  for(int ii = 0; ii < NUMSOCKETS; ii++){
    for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
      std::cout << compute_bound_power[ii][jj] << "\t";
    }
    std::cout << "\n";
  }
#else
#if (defined ERASE_target_energy_method1) || (defined ERASE_target_energy_method2) || (defined ERASE_target_edp_method1) || (defined ERASE_target_edp_method2)
#if (defined DynaDVFS)
  std::string token;
  // Compute-bound Power Models
  infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMAX_MatMul.txt");
  while(std::getline(infile, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        compute_bound_power[0][ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMIN_MatMul.txt");
  while(std::getline(infile1, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        compute_bound_power[3][ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  infile.close();
  infile1.close();
  // Output Power Model profiles
  std::cout << "\nCompute-bound Power Model: \n";
  for(int cc = 0; cc < 4; cc+=3){
    for(int ii = 0; ii < NUMSOCKETS; ii++){
      for (int jj = 0; jj < gotao_nthreads; jj++){
        std::cout << compute_bound_power[cc][ii][jj] << "\t";
      }
      std::cout << "\n";
    }
  }
  // Memory-bound Power Models
  infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMAX_Copy.txt");
  while(std::getline(infile, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        memory_bound_power[0][ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMIN_Copy.txt");
  while(std::getline(infile1, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        memory_bound_power[3][ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  infile.close();
  infile1.close();
  // Output Power Model profiles
  std::cout << "\nMemory-bound Power Model: \n";
  for(int cc = 0; cc < 4; cc+=3){
    for(int ii = 0; ii < NUMSOCKETS; ii++){
      for (int jj = 0; jj < gotao_nthreads; jj++){
        std::cout << memory_bound_power[cc][ii][jj] << "\t";
      }
      std::cout << "\n";
    }
  }
  // Cache-intensive Power Models
  infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMAX_Stencil.txt");
  while(std::getline(infile, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        cache_intensive_power[0][ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMIN_Stencil.txt");
  while(std::getline(infile1, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        cache_intensive_power[3][ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  infile.close();
  infile1.close();
  // Output Power Model profiles
  std::cout << "\nCache-intensive Power Model: \n";
  for(int cc = 0; cc < 4; cc+=3){
    for(int ii = 0; ii < NUMSOCKETS; ii++){
      for (int jj = 0; jj < gotao_nthreads; jj++){
        std::cout << cache_intensive_power[cc][ii][jj] << "\t";
      }
      std::cout << "\n";
    }
  }
#else
  if(denver_freq == 0 && a57_freq == 0){
    infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMAX_MatMul.txt");
    infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMAX_Copy.txt");
    infile2.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMAX_Stencil.txt");
  }
  if(denver_freq == 1 && a57_freq == 0){
    infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMAX_MatMul.txt");
    infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMAX_Copy.txt");
    infile2.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMAX_Stencil.txt");
  }
  if(denver_freq == 0 && a57_freq == 1){
    infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMIN_MatMul.txt");
    infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMIN_Copy.txt");
    infile2.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MAXMIN_Stencil.txt");
  }
  if(denver_freq == 1 && a57_freq == 1){
    infile.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMIN_MatMul.txt");
    infile1.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMIN_Copy.txt");
    infile2.open("/home/nvidia/Desktop/EAS/PowerProfile/TX2_MINMIN_Stencil.txt");
  }
  if(infile.fail() || infile1.fail() || infile2.fail()){
    std::cout << "Failed to open power profile file!" << std::endl;
    std::cin.get();
    return 0;
  }

  std::string token;
  while(std::getline(infile, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        compute_bound_power[ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  // if(Sched == 1){
    // Output the power reading
  std::cout << "Compute-bound Power Model: \n";
  std::cout << "\t Idle \t w=1 \t w=2 \t w=4 \t total\n";
  for(int ii = 0; ii < NUMSOCKETS; ii++){
    if(ii == 0){
      std::cout << "D: ";
    }else{
      std::cout << "A: ";
    }
    for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
      std::cout << compute_bound_power[ii][jj] << "\t";
    }
    std::cout << "\n";
  }
  //}

  while(std::getline(infile1, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        memory_bound_power[ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  // if(Sched == 1){
    // Output the power reading
  std::cout << "\nMemory-bound Power Model: \n";
  for(int ii = 0; ii < NUMSOCKETS; ii++){
    for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
      std::cout << memory_bound_power[ii][jj] << "\t";
    }
    std::cout << "\n";
  }
  // }
  while(std::getline(infile2, token)) {
    std::istringstream line(token);
    int ii = 0;
    int pwidth = 0;
    float get_power = 0;
    while(line >> token) {
      if(ii == 0){
        pwidth = stoi(token);
      }else{
        get_power = stof(token);
        cache_intensive_power[ii-1][pwidth] = get_power; 
      }
      ii++;
    }
  }
  // if(Sched == 1){
    // Output the power reading
  std::cout << "\nCache-intensive Power Model: \n";
  for(int ii = 0; ii < NUMSOCKETS; ii++){
    for (int jj = 0; jj < XITAO_MAXTHREADS; jj++){
      std::cout << cache_intensive_power[ii][jj] << "\t";
    }
    std::cout << "\n";
  }
  // }
#endif
#endif
#endif
#endif


  const char* layout_file = getenv("XITAO_LAYOUT_PATH");
  if(!resources_runtime_conrolled) {
    if(layout_file) {
      int line_count = 0;
      int cluster_count = 0;
      std::string line;      
      std::ifstream myfile(layout_file);
      int current_thread_id = -1; // exclude the first iteration
      if (myfile.is_open()) {
        bool init_affinity = false;
        while (std::getline(myfile,line)) {         
          size_t pos = 0;
          std::string token;
          if(current_thread_id >= XITAO_MAXTHREADS) {
            std::cout << "Fatal error: there are more partitions than XITAO_MAXTHREADS of: " << XITAO_MAXTHREADS  << " in file: " << layout_file << std::endl;    
            exit(0);    
          }
          int thread_count = 0;
          while ((pos = line.find(",")) != std::string::npos) {
            token = line.substr(0, pos);      
            int val = stoi(token);
            if(!init_affinity) static_resource_mapper[thread_count++] = val;  
            else { 
              if(current_thread_id + 1 >= gotao_nthreads) {
                  std::cout << "Fatal error: more configurations than there are input threads in:" << layout_file << std::endl;    
                  exit(0);
              }
              ptt_layout[current_thread_id].push_back(val);
              for(int i = 0; i < val; ++i) {     
                if(current_thread_id + i >= XITAO_MAXTHREADS) {
                  std::cout << "Fatal error: illegal partition choices for thread: " << current_thread_id <<" spanning id: " << current_thread_id + i << " while having XITAO_MAXTHREADS: " << XITAO_MAXTHREADS  << " in file: " << layout_file << std::endl;    
                  exit(0);           
                }
                inclusive_partitions[current_thread_id + i].push_back(std::make_pair(current_thread_id, val)); 
              }              
            }            
            line.erase(0, pos + 1);
          }          
          //if(line_count > 1) {
            token = line.substr(0, line.size());      
            int val = stoi(token);
            if(!init_affinity) static_resource_mapper[thread_count++] = val;
            else { 
              ptt_layout[current_thread_id].push_back(val);
              for(int i = 0; i < val; ++i) {                
                if(current_thread_id + i >= XITAO_MAXTHREADS) {
                  std::cout << "Fatal error: illegal partition choices for thread: " << current_thread_id <<" spanning id: " << current_thread_id + i << " while having XITAO_MAXTHREADS: " << XITAO_MAXTHREADS  << " in file: " << layout_file << std::endl;    
                  exit(0);           
                }
                inclusive_partitions[current_thread_id + i].push_back(std::make_pair(current_thread_id, val)); 
              }              
            }            
          //}
          if(!init_affinity) { 
            gotao_nthreads = thread_count; 
            init_affinity = true;
          }
          current_thread_id++;    
          line_count++;     
          //}
        }
        myfile.close();
      } else {
        std::cout << "Fatal error: could not open hardware layout path " << layout_file << std::endl;    
        exit(0);
      }
    } else {
        std::cout << "Warning: XITAO_LAYOUT_PATH is not set. Default values for affinity and symmetric resoruce partitions will be used" << std::endl;    
        for(int i = 0; i < XITAO_MAXTHREADS; ++i) 
          static_resource_mapper[i] = i; 
        std::vector<int> widths;             
        int count = gotao_nthreads;        
        std::vector<int> temp;        // hold the big divisors, so that the final list of widths is in sorted order 
        for(int i = 1; i < sqrt(gotao_nthreads); ++i){ 
          if(gotao_nthreads % i == 0) {
            widths.push_back(i);
            temp.push_back(gotao_nthreads / i); 
          } 
        }
        std::reverse(temp.begin(), temp.end());
        widths.insert(widths.end(), temp.begin(), temp.end());
        //std::reverse(widths.begin(), widths.end());        
        for(int i = 0; i < widths.size(); ++i) {
          for(int j = 0; j < gotao_nthreads; j+=widths[i]){
            ptt_layout[j].push_back(widths[i]);
          }
        }

        for(int i = 0; i < gotao_nthreads; ++i){
          for(auto&& width : ptt_layout[i]){
            for(int j = 0; j < width; ++j) {                
              inclusive_partitions[i + j].push_back(std::make_pair(i, width)); 
            }         
          }
        }
      } 
  } else {    
    if(gotao_nthreads != runtime_resource_mapper.size()) {
      std::cout << "Warning: requested " << runtime_resource_mapper.size() << " at runtime, whereas gotao_nthreads is set to " << gotao_nthreads <<". Runtime value will be used" << std::endl;
      gotao_nthreads = runtime_resource_mapper.size();
    }            
  }
#ifdef DEBUG
	std::cout << "XiTAO initialized with " << gotao_nthreads << " threads and configured with " << XITAO_MAXTHREADS << " max threads " << std::endl;
  // std::cout << "The platform has " << cluster_mapper.size() << " clusters.\n";
  // for(int i = 0; i < cluster_mapper.size(); i++){
  //   std::cout << "[DEBUG] Cluster " << i << " has " << cluster_mapper[i] << " cores.\n";
  // }
  for(int i = 0; i < static_resource_mapper.size(); ++i) {
    std::cout << "[DEBUG] Thread " << i << " is configured to be mapped to core id : " << static_resource_mapper[i] << std::endl;
    std::cout << "[DEBUG] PTT Layout Size of thread " << i << " : " << ptt_layout[i].size() << std::endl;
    std::cout << "[DEBUG] Inclusive partition size of thread " << i << " : " << inclusive_partitions[i].size() << std::endl;
    std::cout << "[DEBUG] leader thread " << i << " has partition widths of : ";
    for (int j = 0; j < ptt_layout[i].size(); ++j){
      std::cout << ptt_layout[i][j] << " ";
    }
    std::cout << std::endl;
    std::cout << "[DEBUG] thread " << i << " is contained in these [leader,width] pairs : ";
    for (int j = 0; j < inclusive_partitions[i].size(); ++j){
      std::cout << "["<<inclusive_partitions[i][j].first << "," << inclusive_partitions[i][j].second << "]";
    }
    std::cout << std::endl;
  }
#endif

  if(nhwc>=0){
    gotao_ncontexts = nhwc;
  }
  else{
    if(getenv("GOTAO_HW_CONTEXTS")){
      gotao_ncontexts = atoi(getenv("GOTAO_HW_CONTEXTS"));
    }
    else{ 
      gotao_ncontexts = GOTAO_HW_CONTEXTS;
    }
  }

#if defined(Haswell)
  if(nhwc >= 0){
    num_sockets = nhwc;
  }
  else{
    if(getenv("NUMSOCKETS")){
      num_sockets = atoi(getenv("NUMSOCKETS"));
    }
    else{
      num_sockets = NUMSOCKETS;
    }
  } 
#endif

  if(thrb>=0){
    gotao_thread_base = thrb;
  }
  else{
    if(getenv("GOTAO_THREAD_BASE")){
      gotao_thread_base = atoi(getenv("GOTAO_THREAD_BASE"));
    }
    else{
      gotao_thread_base = GOTAO_THREAD_BASE;
    }
  }
/*
  starting_barrier = new BARRIER(gotao_nthreads + 1);
  tao_barrier = new cxx_barrier(2);
  for(int i = 0; i < gotao_nthreads; i++){
    t[i]  = new std::thread(worker_loop, i);   
  }
*/  
}

// Initialize gotao from environment vars or defaults
int gotao_init(int scheduler, int numkernels, int STEAL_DtoA, int STEAL_AtoD){
  starting_barrier = new BARRIER(gotao_nthreads);
  tao_barrier = new cxx_barrier(gotao_nthreads);
  for(int i = 0; i < gotao_nthreads; i++){
    t[i]  = new std::thread(worker_loop, i);
  }
  Sched = scheduler;
  num_kernels = numkernels;
  maySteal_DtoA = STEAL_DtoA;
  maySteal_AtoD = STEAL_AtoD;
#ifdef DynaDVFS
  current_freq = 2035200; // ERASE starting frequency is 2.04GHz for both clusters
  env = 0;
#endif
}

int gotao_start()
{
  /*
  if(Sched == 0){
  //Analyse DAG based on tasks in ready q and asign criticality values
  for(int j=0; j<gotao_nthreads; j++){
    //Iterate over all ready tasks for all threads
    for(std::list<PolyTask *>::iterator it = worker_ready_q[j].begin(); it != worker_ready_q[j].end(); ++it){
      //Call recursive function setting criticality
      (*it)->set_criticality();
    }
  }
  for(int j = 0; j < gotao_nthreads; j++){
    for(std::list<PolyTask *>::iterator it = worker_ready_q[j].begin(); it != worker_ready_q[j].end(); ++it){
      if ((*it)->criticality == critical_path){
        (*it)->marker = 1;
        (*it) -> set_marker(1);
        break;
      }
    }
  }
  }
  */
  starting_barrier->wait(gotao_nthreads+1);
}

int gotao_fini()
{
  resources_runtime_conrolled = false;
  gotao_can_exit = true;
  gotao_initialized = false;
  for(int i = 0; i < gotao_nthreads; i++){
    t[i]->join();
  }
}

void gotao_barrier()
{
  tao_barrier->wait();
}

int check_and_get_available_queue(int queue) {
  bool found = false;
  if(queue >= runtime_resource_mapper.size()) {
    return rand()%runtime_resource_mapper.size();
  } else {
    return queue;
  }  
}
// push work into polytask queue
// if no particular queue is specified then try to determine which is the local
// queue and insert it there. This has some overhead, so in general the
// programmer should specify some queue
int gotao_push(PolyTask *pt, int queue)
{
  if((queue == -1) && (pt->affinity_queue != -1)){
    queue = pt->affinity_queue;
  }
  else{
    if(queue == -1){
      queue = sched_getcpu();
    }
  }
  if(resources_runtime_conrolled) queue = check_and_get_available_queue(queue);
  LOCK_ACQUIRE(worker_lock[queue]);
  worker_ready_q[queue].push_front(pt);
  LOCK_RELEASE(worker_lock[queue]);
}

// Push work when not yet running. This version does not require locks
// Semantics are slightly different here
// 1. the tid refers to the logical core, before adjusting with gotao_thread_base
// 2. if the queue is not specified, then put everything into the first queue
int gotao_push_init(PolyTask *pt, int queue)
{
  if((queue == -1) && (pt->affinity_queue != -1)){
    queue = pt->affinity_queue;
  }
  else{
    if(queue == -1){
      queue = gotao_thread_base;
    }
  }
  if(resources_runtime_conrolled) queue = check_and_get_available_queue(queue);
  worker_ready_q[queue].push_front(pt);
}

// alternative version that pushes to the back
int gotao_push_back_init(PolyTask *pt, int queue)
{
  if((queue == -1) && (pt->affinity_queue != -1)){
    queue = pt->affinity_queue;
  }
  else{
    if(queue == -1){
      queue = gotao_thread_base;
    }
  }
  worker_ready_q[queue].push_back(pt);
}


long int r_rand(long int *s)
{
  *s = ((1140671485*(*s) + 12820163) % (1<<24));
  return *s;
}


void __xitao_lock()
{
  smpd_region_lock.lock();
  //LOCK_ACQUIRE(smpd_region_lock);
}
void __xitao_unlock()
{
  smpd_region_lock.unlock();
  //LOCK_RELEASE(smpd_region_lock);
}

int worker_loop(int nthread){
  // pmc.open("PMC.txt", std::ios_base::app);
  // std::ofstream timetask;
  // timetask.open("data_process.sh", std::ios_base::app);

  int phys_core;
  if(resources_runtime_conrolled) {
    if(nthread >= runtime_resource_mapper.size()) {
      LOCK_ACQUIRE(output_lck);
      std::cout << "Error: thread cannot be created due to resource limitation" << std::endl;
      LOCK_RELEASE(output_lck);
      exit(0);
    }
    phys_core = runtime_resource_mapper[nthread];
  } else {
    phys_core = static_resource_mapper[gotao_thread_base+(nthread%(XITAO_MAXTHREADS-gotao_thread_base))];   
  }
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] nthread: " << nthread << " mapped to physical core: "<< phys_core << std::endl;
  LOCK_RELEASE(output_lck);
#endif  
  unsigned int seed = time(NULL);
  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  CPU_SET(phys_core, &cpu_mask);

  sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask); 
  // When resources are reclaimed, this will preempt the thread if it has no work in its local queue to do.
  
  PolyTask *st = nullptr;
  starting_barrier->wait(gotao_nthreads+1);  
  auto&&  partitions = inclusive_partitions[nthread];

  // Perf Event Counters
  struct perf_event_attr pea;
  int fd1;
  uint64_t id1, val1;
  char buf[4096];
  struct read_format* rf = (struct read_format*) buf;

  memset(&pea, 0, sizeof(struct perf_event_attr));
  pea.type = PERF_TYPE_HARDWARE;
  pea.size = sizeof(struct perf_event_attr);
  pea.config = PERF_COUNT_HW_CPU_CYCLES;
  pea.disabled = 1;
  pea.exclude_kernel = 0;
  pea.exclude_hv = 1;
  pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  fd1 = syscall(__NR_perf_event_open, &pea, 0, phys_core, -1, 0);
  ioctl(fd1, PERF_EVENT_IOC_ID, &id1);


  if (!Denver.is_open() || !ARM.is_open()){
    std::cerr << "[Child] Somthing failed while opening the file! " << std::endl;
    return 0;
  }

  int idle_try = 0;
  int idle_times = 0;
  int SleepNum = 0;
  int AccumTime = 0;

  if(Sched == 1){
    for(int i=0; i<XITAO_MAXTHREADS; i++){ 
      status[i] = 1;
      status_working[i] = 1;
    }
  }
  bool stop = false;

  // Accumulation of tasks execution time
  // Goal: Get runtime idle time
#ifdef EXECTIME
  //std::chrono::time_point<std::chrono::system_clock> idle_start, idle_end;
  std::chrono::duration<double> elapsed_exe;
#endif

#ifdef OVERHEAD_PTT
  std::chrono::duration<double> elapsed_ptt;
#endif

#ifdef PTTaccuracy
  float MAE = 0.0f;
  std::ofstream PTT("PTT_Accuracy.txt");
#endif

#ifdef Energyaccuracy
  float EnergyPrediction = 0.0f;
#endif

  while(true)
  {    
    int random_core = 0;
    AssemblyTask *assembly = nullptr;
    SimpleTask *simple = nullptr;

  // 0. If a task is already provided via forwarding then exeucute it (simple task)
  //    or insert it into the assembly queues (assembly task)
    if( st && !stop){
      if(st->type == TASK_SIMPLE){
        SimpleTask *simple = (SimpleTask *) st;
        simple->f(simple->args, nthread);

  #ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Distributing simple task " << simple->taskid << " with width " << simple->width << " to workers " << nthread << std::endl;
        LOCK_RELEASE(output_lck);
  #endif
  #ifdef OVERHEAD_PTT
        st = simple->commit_and_wakeup(nthread, elapsed_ptt);
  #else
        st = simple->commit_and_wakeup(nthread);
  #endif
        simple->cleanup();
        //delete simple;
      }
      else 
      if(st->type == TASK_ASSEMBLY){
        AssemblyTask *assembly = (AssemblyTask *) st;
#if defined(TX2)
        if(Sched == 3){
          assembly->leader = nthread / assembly->width * assembly->width; // homogenous calculation of leader core
        }
#endif
#if defined(Haswell) || defined(CATS)
        assembly->leader = nthread / assembly->width * assembly->width;
#endif
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Distributing " << assembly->kernel_name << " task " << assembly->taskid << " with width " << assembly->width << " to workers [" << assembly->leader << "," << assembly->leader + assembly->width << ")" << std::endl;
        LOCK_RELEASE(output_lck);
#endif
        // std::cout << "Task: " << assembly->kernel_name << "\n";
        /* After getting the best config, and before distributing to AQs: 
        (1) Coarse-grained task: check if it is needed to tune the frequency; 
        (2) Fine-grained task: check the WQs of the cluster include N consecutive same tasks, that the total execution time of these N tasks > threshold, then search for the best frequency and then tune the frequency */
        if(global_training == true && assembly->get_bestconfig_state() == true){
          if(assembly->granularity_fine == true && assembly->get_enable_freq_change() == false){ /* (2) Fine-grained tasks && not allowed to do frequency change currently */
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Assembly Task " << assembly->taskid << " is a fine-grained task. " << std::endl;
          LOCK_RELEASE(output_lck);
#endif 
          int best_cluster = assembly->get_best_cluster();
          int best_width = assembly->get_best_numcores();
          // int travel_fine_grained = 0;
          int consecutive_fine_grained = 1; /* assembly already is one fine-grained task, so initilize to 1 */
          for(int i = 0; i < 8; i++){ /* Assume the maximum of each work queue size is 8 */
            for(int j = start_coreid[best_cluster]; j < end_coreid[best_cluster]; j++){
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] Visit the " << i << "th task in the queues, visit the " << j << "th core in cluster " << best_cluster << std::endl;
              LOCK_RELEASE(output_lck);
#endif 
              if(worker_ready_q[j].size() > 0){
                std::list<PolyTask *>::iterator it = worker_ready_q[j].begin();
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] The queue size of " << j << "th core > 0. Now point to the begining task. " << std::endl;
                LOCK_RELEASE(output_lck);
#endif 
                if(i > 0){
                  std::advance(it, i);
                }
                if((*it)->granularity_fine == true && (*it)->tasktype == assembly->tasktype){
                  consecutive_fine_grained++; /* Another same type of task */
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] The pointing task *it is also a same type + fine-grained task as the assembly task. consecutive_fine_grained = " << consecutive_fine_grained << std::endl;
                  LOCK_RELEASE(output_lck);
#endif 
                  if(consecutive_fine_grained * assembly->get_timetable(cur_freq_index[best_cluster], best_cluster, best_width-1) * (end_coreid[best_cluster] - start_coreid[best_cluster]) / best_width > FINE_GRAIN_THRESHOLD){
                    // find out the best frequency here
#ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] Enough same type of tasks! Find out the best frequency now!" << std::endl;
                    LOCK_RELEASE(output_lck);
#endif                    
                    float idleP_cluster = 0.0f;
                    float shortest_exec = 100000.0f;
                    float energy_pred = 0.0f;
                    int sum_cluster_active = std::accumulate(status+start_coreid[1-best_cluster], status+end_coreid[1-best_cluster], 0); /* If there is any active cores in another cluster */
#ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] Number of active cores in cluster " << 1-best_cluster << ": " << sum_cluster_active << ". status[0] = " << status[0] \
                    << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] << std::endl;
                    LOCK_RELEASE(output_lck);
#endif 
                    for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                      if(sum_cluster_active == 0){ /* the number of active cores is zero in another cluster */
                        idleP_cluster = idle_power[freq_indx][best_cluster] + idle_power[freq_indx][1-best_cluster]; /* Then equals idle power of whole chip */
#ifdef DEBUG
                        LOCK_ACQUIRE(output_lck);
                        std::cout << "[DEBUG] Cluster " << 1-best_cluster << " no active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
                        LOCK_RELEASE(output_lck);
#endif 
                      }else{
                        idleP_cluster = idle_power[freq_indx][best_cluster]; /* otherwise, equals idle power of the cluster */
#ifdef DEBUG
                        LOCK_ACQUIRE(output_lck);
                        std::cout << "[DEBUG] Cluster " << 1-best_cluster << " has active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
                        LOCK_RELEASE(output_lck);
#endif 
                      }
                      sum_cluster_active = (sum_cluster_active < best_width)? best_width : sum_cluster_active;
                      float idleP = idleP_cluster * best_width / sum_cluster_active;
                      float runtimeP = assembly->get_powertable(freq_indx, best_cluster, best_width-1);
                      float timeP = assembly->get_timetable(freq_indx, best_cluster, best_width-1);
                      energy_pred = timeP * (runtimeP + idleP); /* assembly->get_powertable() is only the prediction for runtime power. */
// #ifdef DEBUG
                      LOCK_ACQUIRE(output_lck);
                      std::cout << "[DEBUG] For the fine-grained tasks, frequency: " << avail_freq[freq_indx] << " on cluster " << best_cluster << " with width "<< best_width << ", idle power " << idleP << ", runtime power " \
                      << runtimeP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
                      LOCK_RELEASE(output_lck);
// #endif 
                      if(energy_pred < shortest_exec){
                        shortest_exec = energy_pred;
                        assembly->set_best_freq(freq_indx);
                      }
                    }
// #ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] For the fine-grained tasks, get the optimal frequency: " << avail_freq[assembly->get_best_freq()] << std::endl;
                    LOCK_RELEASE(output_lck);
// #endif 
                    goto consecutive_true; // No more searching
                  }else{
                    continue;
                  }
                }else{
                  goto consecutive_false;
                }
              }else{
                continue;
              } 
            }
          }
          consecutive_true: assembly->set_enable_freq_change(true); /* TBD Problem: should mark traversed tasks, next following tasks should do the process again, since the DAG might include other type of tasks */
          consecutive_false:;
        }
          /*Tune frequency if required != current, for both fine-grained and coarse-grained tasks */
          if(assembly->get_enable_freq_change() == true){ /* Allow to do frequency tuning for the tasks */
            assembly->bestfreq = avail_freq[assembly->get_best_freq()];
            int best_cluster = assembly->get_best_cluster();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " is allowed to do frequency scaling. \n";
            LOCK_RELEASE(output_lck);
#endif
            if(assembly->bestfreq != cur_freq[best_cluster]){ /* check if the required frequency equals the current frequency! */
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << ", Current frequency != required frequency. Change it! \n";
              LOCK_RELEASE(output_lck);
#endif
              if(best_cluster == 0){ //Denver
                Denver << std::to_string(assembly->bestfreq) << std::endl;
                /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
                int cluster_active = std::accumulate(status_working + start_coreid[1], status_working + end_coreid[1], 0);   
                if(cluster_active == 0 && cur_freq_index[1] > cur_freq_index[0]){ /* No working cores on A57 cluster and the current frequency of A57 is higher than working Denver, then tune the frequency */
                  ARM << std::to_string(assembly->bestfreq) << std::endl;
                  cur_freq[1] = assembly->bestfreq; /* Update the current frequency */
                  cur_freq_index[1] = assembly->get_best_freq();
                }  
              }else{
                ARM << std::to_string(assembly->bestfreq) << std::endl;
                /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
                int cluster_active = std::accumulate(status_working + start_coreid[0], status_working + end_coreid[0], 0);   
                if(cluster_active == 0 && cur_freq_index[0] > cur_freq_index[1]){ /* No working cores on Denver cluster and the current frequency of Denver is higher than working A57, then tune the frequency */
                  Denver << std::to_string(assembly->bestfreq) << std::endl;
                  cur_freq[0] = assembly->bestfreq; /* Update the current frequency */
                  cur_freq_index[0] = assembly->get_best_freq();
                }  
              }
              cur_freq[best_cluster] = assembly->bestfreq; /* Update the current frequency */
              cur_freq_index[best_cluster] = assembly->get_best_freq(); /* Update the current frequency index */
            }
          }
        }
//           for(std::list<PolyTask *>::iterator it = worker_ready_q[start_coreid[best_cluster]].begin(); it != worker_ready_q[start_coreid[best_cluster]].end(); ++it){
//             if((*it)->granularity_fine == true && (*it)->tasktype == assembly->tasktype){
//               consecutive_fine_grained++; /* Another same type of task */
//               if(consecutive_fine_grained * assembly->get_timetable(cur_freq_index[0], 0, 1) > FINE_GRAIN_THRESHOLD){ /* Check if it is worth to change frequency or not by using the execution time of 2 Denver. !!! 2 Denver? !!! */
//                 /* Find out the best frequency for the kernel task here */
//                 float idleP_cluster = 0.0f;
//                 float shortest_exec = 100000.0f;
//                 float energy_pred = 0.0f;
//                 int sum_cluster_active = std::accumulate(status+start_coreid[1-best_cluster], status+end_coreid[1-best_cluster], 0); /* If there is any active cores in another cluster */
// #ifdef DEBUG
//                 LOCK_ACQUIRE(output_lck);
//                 std::cout << "[DEBUG] Number of active cores in cluster " << 1-best_cluster << ": " << sum_cluster_active << ". status[0] = " << status[0] \
//                 << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] << std::endl;
//                 LOCK_RELEASE(output_lck);
// #endif 
//                 for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
//                   if(sum_cluster_active == 0){ /* the number of active cores is zero in another cluster */
//                     idleP_cluster = idle_power[freq_indx][best_cluster] + idle_power[freq_indx][1-best_cluster]; /* Then equals idle power of whole chip */
// #ifdef DEBUG
//                     LOCK_ACQUIRE(output_lck);
//                     std::cout << "[DEBUG] Cluster " << 1-best_cluster << " no active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
//                     LOCK_RELEASE(output_lck);
// #endif 
//                   }else{
//                     idleP_cluster = idle_power[freq_indx][best_cluster]; /* otherwise, equals idle power of the cluster */
// #ifdef DEBUG
//                     LOCK_ACQUIRE(output_lck);
//                     std::cout << "[DEBUG] Cluster " << 1-best_cluster << " has active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
//                     LOCK_RELEASE(output_lck);
// #endif 
//                   }
//                   sum_cluster_active = (sum_cluster_active < best_width)? best_width : sum_cluster_active;
//                   float idleP = idleP_cluster * best_width / sum_cluster_active;
//                   float runtimeP = assembly->get_powertable(freq_indx, best_cluster, best_width-1);
//                   float timeP = assembly->get_timetable(freq_indx, best_cluster, best_width-1);
//                   energy_pred = timeP * (runtimeP + idleP); /* assembly->get_powertable() is only the prediction for runtime power. */
// // #ifdef DEBUG
//                   LOCK_ACQUIRE(output_lck);
//                   std::cout << "[DEBUG] For the fine-grained tasks, frequency: " << avail_freq[freq_indx] << " on cluster " << best_cluster << " with width "<< best_width << ", idle power " << idleP << ", runtime power " \
//                   << runtimeP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
//                   LOCK_RELEASE(output_lck);
// // #endif 
//                   if(energy_pred < shortest_exec){
//                     shortest_exec = energy_pred;
//                     assembly->set_best_freq(freq_indx);
//                   }
//                 }
// // #ifdef DEBUG
//                 LOCK_ACQUIRE(output_lck);
//                 std::cout << "[DEBUG] For the fine-grained tasks, get the optimal frequency: " << avail_freq[assembly->get_best_freq()] << std::endl;
//                 LOCK_RELEASE(output_lck);
// // #endif 
//                 /*Tune frequency if required != current */
//                 assembly->bestfreq = assembly->get_best_freq();
//                 if(assembly->bestfreq != cur_freq[best_cluster]){ /* check if the required frequency equals the current frequency! */
//                   if(best_cluster == 0){ //Denver
//                     Denver << std::to_string(assembly->bestfreq) << std::endl;
//                     /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
//                     int cluster_active = std::accumulate(status_working + start_coreid[1], status_working + end_coreid[1], 0);   
//                     if(cluster_active == 0 && cur_freq_index[1] > cur_freq_index[0]){ /* No working cores on A57 cluster and the current frequency of A57 is higher than working Denver, then tune the frequency */
//                       ARM << std::to_string(assembly->bestfreq) << std::endl;
//                       cur_freq[1] = assembly->bestfreq; /* Update the current frequency */
//                     }  
//                   }else{
//                     ARM << std::to_string(assembly->bestfreq) << std::endl;
//                     /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
//                     int cluster_active = std::accumulate(status_working + start_coreid[0], status_working + end_coreid[0], 0);   
//                     if(cluster_active == 0 && cur_freq_index[0] > cur_freq_index[1]){ /* No working cores on Denver cluster and the current frequency of Denver is higher than working A57, then tune the frequency */
//                       Denver << std::to_string(assembly->bestfreq) << std::endl;
//                       cur_freq[0] = assembly->bestfreq; /* Update the current frequency */
//                     }  
//                   }
//                   cur_freq[best_cluster] = assembly->bestfreq; /* Update the current frequency */
//                 }
//                 break; // No more searching
//               }else{
//                 for(int j = start_coreid[best_cluster] + 1; j < end_coreid[best_cluster]; j++){
//                   std::list<PolyTask *>::iterator itn = worker_ready_q[j].begin() + travel_fine_grained;
//                   if((*itn)->granularity_fine == true && (*itn)->tasktype == assembly->tasktype){
//                     consecutive_fine_grained++; /* Another same type of task */
//                     if(consecutive_fine_grained * assembly->get_timetable(cur_freq_index[0], 0, 1) > FINE_GRAIN_THRESHOLD){
//                       // find out the best frequency here

//                     }else{
//                       continue;
//                     }
//                   }else{
//                     break;
//                   }
//                 }
//               }
//               continue;
//             }else{
//               break;
//             }
//             travel_fine_grained++;
//           }
        for(int i = assembly->leader; i < assembly->leader + assembly->width; i++){
          LOCK_ACQUIRE(worker_assembly_lock[i]);
          worker_assembly_q[i].push_back(st);
#ifdef NUMTASKS_MIX
#ifdef ONLYCRITICAL
          int pr = assembly->if_prio(nthread, assembly);
          if(pr == 1){
#endif
            num_task[assembly->tasktype][ assembly->width * gotao_nthreads + i]++;
#ifdef ONLYCRITICAL
          }
#endif
#endif
        }
        for(int i = assembly->leader; i < assembly->leader + assembly->width; i++){
          LOCK_RELEASE(worker_assembly_lock[i]);
        }
        st = nullptr;
      }
      continue;
    }

    // 1. check for assemblies
    if(!worker_assembly_q[nthread].pop_front(&st)){
      st = nullptr;
    }
  // assemblies are inlined between two barriers
    if(st) {
      int _final = 0; // remaining
      assembly = (AssemblyTask *) st;
#ifdef NEED_BARRIER
      if(assembly->width > 1){
        assembly->barrier->wait(assembly->width);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout <<"[BARRIER-Before] For Task " << assembly->taskid << ", thread "<< nthread << " arrives." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
      }
#endif
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] Thread "<< nthread << " starts executing " << assembly->kernel_name << " task " << assembly->taskid << "......\n";
      LOCK_RELEASE(output_lck);
#endif
      // if(Sched == 1 && nthread == assembly->leader){
      if(Sched == 1){
#if defined Performance_Model_Cycle 
        ioctl(fd1, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fd1, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
        int clus_id = (nthread < START_A)? 0:1;
        assembly->start_running_freq = cur_freq[clus_id];
      }

      std::chrono::time_point<std::chrono::system_clock> t1,t2;
      t1 = std::chrono::system_clock::now();
      auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t1);
      auto epoch1 = start1_ms.time_since_epoch();
      
      assembly->execute(nthread);

      t2 = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = t2-t1;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " execution time on thread " << nthread << ": " << elapsed_seconds.count() << "\n";
      LOCK_RELEASE(output_lck);
#endif 
      // if(Sched == 1 && nthread == assembly->leader){
#if defined Performance_Model_Cycle 
      if(Sched == 1){
        ioctl(fd1, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        read(fd1, buf, sizeof(buf));
        for (int i = 0; i < rf->nr; i++){
          if (rf->values[i].id == id1) {
              val1 = rf->values[i].value;
          }
        }
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " execution time on thread " << nthread << ": " << elapsed_seconds.count() << ", cycles: " << val1 << "\n";
        LOCK_RELEASE(output_lck);
#endif 
      }
#endif
#ifdef PowerProfiling
      auto end1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t2);
      auto epoch1_end = end1_ms.time_since_epoch();
      out << assembly->kernel_name << "\t" << epoch1.count() << "\t" << epoch1_end.count() << "\n";
      out.flush();
#endif
#ifdef NEED_BARRIER
      if(assembly->width > 1){
        assembly->barrier->wait(assembly->width);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout <<"[BARRIER-After] For Task " << assembly->taskid << " thread  "<< nthread << " arrives." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
      }
#endif
#ifdef EXECTIME
      elapsed_exe += t2-t1;
#endif

      if(Sched == 1 && assembly->get_timetable_state(2) == false && assembly->tasktype < num_kernels){  /* Only leader core update the PTT entries */
        double ticks = elapsed_seconds.count();       
        int width_index = assembly->width - 1;
        /* (1) Was running on Denver (2) Denver PTT table hasn't finished training     (3) if the frequency changing is happening during the task execution, do not update the table*/
        if(assembly->leader < START_A){
        if(assembly->get_timetable_state(0) == false && assembly->start_running_freq == cur_freq[0]){
          /* Step 1: Update PTT table values */
          if(ptt_freq_index[0] == 0){ /*2.04GHz*/ 
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, 0, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = float(std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0)) / float(assembly->width);
                if(oldticks == 0.0f || newticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 0, newticks, width_index);
                }
              }else{
                if(oldticks == 0.0f || ticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 0, ticks, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(0, 0, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[2.04GHz, Denver, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(0,0, width_index) << ". Current time: " << assembly->get_timetable(0, 0, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
//             float oldticks = assembly->get_timetable(0, 0, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
//               assembly->set_timetable(0, 0, ticks, width_index);
// #if defined Performance_Model_Cycle   
//               assembly->set_cycletable(0, 0, val1, width_index);
// #endif
//             }
//             // else{
//               // assembly->set_timetable(0, 0, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             assembly->increment_PTT_UpdateFinish(0, 0, width_index);
//             // }
          }else{ /*1.11GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 0, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || newticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(NUM_AVAIL_FREQ/2, 0, newticks, width_index);
                }
              }else{
                if(oldticks == 0.0f || ticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(NUM_AVAIL_FREQ/2, 0, ticks, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(1, 0, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[1.11GHz, Denver, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(1,0, width_index) << ". Current time: " << assembly->get_timetable(NUM_AVAIL_FREQ/2, 0, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
//             float oldticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 0, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){
//               assembly->set_timetable(NUM_AVAIL_FREQ/2, 0, ticks, width_index); 
// #if defined Performance_Model_Cycle  
//               assembly->set_cycletable(1, 0, val1, width_index);
// #endif
//             }
//             // else{
//             //   assembly->set_timetable(NUM_AVAIL_FREQ/2, 0, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             // assembly->PTT_UpdateFinish[1][0][width_index]++;
//             assembly->increment_PTT_UpdateFinish(1, 0, width_index);
//             // }
          }
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_UpdateFinish[" << ptt_freq_index[0] << ", Denver, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(ptt_freq_index[0],0, width_index) << ".\n";
//           LOCK_RELEASE(output_lck);
// #endif
//           /* Step 2: Update cycle table values */
//           uint64_t oldcycles = assembly->get_cycletable(ptt_freq_index[0], 0, width_index);
//           if(oldcycles == 0){
//             assembly->set_cycletable(ptt_freq_index[0], 0, val1, width_index);
//           }else{
//             assembly->set_cycletable(ptt_freq_index[0], 0, ((oldcycles + val1)/2), width_index);
//           }
//           if(assembly->get_timetable_state(0) == false){ /* PTT training hasn't finished */   

            if (ptt_freq_index[0] == 0){          /* Current frequency is 2.04GHz */
              if(assembly->get_PTT_UpdateFinish(0, 0, 0) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 0, 1) >= NUM_TRAIN_TASKS){ /* First dimention: 2.04GHz, second dimention: Denver, third dimention: width_index */
                PTT_finish_state[0][0][assembly->tasktype] = 1; /* First dimention: 2.04GHz, second dimention: Denver, third dimention: tasktype */
                if(std::accumulate(std::begin(PTT_finish_state[0][0]), std::begin(PTT_finish_state[0][0])+num_kernels, 0) == num_kernels){ /* Check all kernels have finished the training of Denver at 2.04GHz */
                  Denver << std::to_string(1113600) << std::endl;
                  ptt_freq_index[0] = 1;
                  cur_freq[0] = 1113600;
                  cur_freq_index[0] = NUM_AVAIL_FREQ/2;
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] Denver cluster completed PTT training at 2.04GHz, turn to 1.11GHz. \n";
                  LOCK_RELEASE(output_lck);
#endif
                }
              }
//               int ptt_check = 0; /* Check if PTT values of Denver are filled out */
//               for(auto&& width : ptt_layout[START_D]) { 
//                 float check_ticks = assembly->get_timetable(0, 0, width - 1);  /* First parameter is ptt_freq_index[0] = 0 */
// #if defined Performance_Model_Cycle
//                 uint64_t check_cycles = assembly->get_cycletable(0, 0, width - 1); 
//                 if(assembly->get_PTT_UpdateFinish(0, 0, width-1) >= NUM_TRAIN_TASKS && check_ticks > 0.0f && check_cycles > 0)
// #endif
// #if defined Performance_Model_Time
//                 if(assembly->get_PTT_UpdateFinish(0, 0, width-1) >= NUM_TRAIN_TASKS && check_ticks > 0.0f)
// #endif
//                 {
//                   ptt_check++;
// #ifdef DEBUG
//                   LOCK_ACQUIRE(output_lck);
//                   std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[2.04GHz, Denver, " << width << "] = " << check_ticks << ". ptt_check = " << ptt_check << ".\n";
//                   LOCK_RELEASE(output_lck);
// #endif
//                   continue;
//                 }else{
//                   break;
//                 }
//               }
//               if(ptt_check == num_width[0]){ /* All ptt values of Denver are positive => visited at least once, then configure the frequency to 1.11GHz */
//                 ptt_freq_index[0] = 1;
//                 Denver << std::to_string(1113600) << std::endl;
//                 cur_freq[0] = 1113600;
//                 cur_freq_index[0] = NUM_AVAIL_FREQ/2;
// #ifdef DEBUG
//                 LOCK_ACQUIRE(output_lck);
//                 std::cout << "[DEBUG] Denver cluster completed PTT training at 2.04GHz, turn to 1.11GHz. \n";
//                 LOCK_RELEASE(output_lck);
// #endif
//               }             
            }else{ /* Current frequency is 1.11GHz */
              int ptt_check = 0;            
              for(auto&& width : ptt_layout[START_D]){ 
                float check_ticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 0, width - 1); /* First parameter is ptt_freq_index[0] = 12/2 = 6, 1.11GHz */
                if(assembly->get_PTT_UpdateFinish(1, 0, width-1) >= NUM_TRAIN_TASKS && check_ticks > 0.0f){ /* PTT_UpdateFinish first dimention is 1 => 1.11GHz */
                  ptt_check++;
                  if(assembly->get_mbtable(0, width-1) == 0.0f){ // If the memory-boundness of this config hasn't been computed yet
                    float memory_boundness = 0.0f;
#if defined Performance_Model_Cycle                  /* Method 1: Calculate Memory-boundness using cycles = (1 - cycle2/cycle1) / (1- f2/f1) */
                    uint64_t check_cycles = assembly->get_cycletable(1, 0, width - 1); /* get_cycletable first dimention is 1 => 1.11GHz */
                    uint64_t cycles_high = assembly->get_cycletable(0, 0, width-1); /* First parameter is 0, means 2.04GHz, check_cycles is at 1.11Ghz */
                    float a = 1 - float(check_cycles) / float(cycles_high);
                    float b = 1 - float(avail_freq[NUM_AVAIL_FREQ/2]) / float(avail_freq[0]);
                    memory_boundness = a/b;
#endif
#if defined Performance_Model_Time                   /* Method 2: Calculate Memory-boundness (using execution time only) = ((T2f2/T1)-f1) / (f2-f1) */
                    float highest_ticks = assembly->get_timetable(0, 0, width - 1);
                    float a = float(avail_freq[0]) / float(avail_freq[NUM_AVAIL_FREQ/2]);
                    float b = check_ticks / highest_ticks;
                    memory_boundness = (b-a) / (1-a);
                    LOCK_ACQUIRE(output_lck);
                    std::cout << assembly->kernel_name << ": Memory-boundness Calculation (Denver, width " << width << ") = " << memory_boundness << ". a = " << a << ", b = " << b << std::endl;
                    LOCK_RELEASE(output_lck);
#endif
                    assembly->set_mbtable(0, memory_boundness, width-1); /*first parameter: cluster 0 - Denver, second parameter: update value, third value: width_index */
                    if(memory_boundness > 1){
                      LOCK_ACQUIRE(output_lck);
                      std::cout << "[Warning]" << assembly->kernel_name << "->Memory-boundness (Denver) is greater than 1!" << std::endl;
                      LOCK_RELEASE(output_lck);
                      memory_boundness = 1;
                      for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
#if defined Performance_Model_Cycle 
                        uint64_t new_cycles = cycles_high * float(avail_freq[freq_indx])/float(avail_freq[0]);
                        float ptt_value_newfreq = float(new_cycles) / float(avail_freq[freq_indx]*1000);
#endif
#if defined Performance_Model_Time
                        float ptt_value_newfreq = highest_ticks;
#endif
                        assembly->set_timetable(freq_indx, 0, ptt_value_newfreq, width-1);
                      }
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* (2) Power value prediction */
                          assembly->set_powertable(freq_indx, 0, runtime_power[9][freq_indx][1][width-1], width-1); 
                      }
                    }else{
                      if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
                        memory_boundness = 0.0001;
                        for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* Fill out PTT[other freqs]. Start from 1, which is next freq after 2.04, also skip 1.11GHz*/
                          if(freq_indx == NUM_AVAIL_FREQ/2){
                            continue;
                          }else{
#if defined Performance_Model_Cycle 
                            float ptt_value_newfreq = float(cycles_high) / float(avail_freq[freq_indx]*1000); /* (1) Execution time prediction */
#endif
#if defined Performance_Model_Time
                            float ptt_value_newfreq = highest_ticks * (float(avail_freq[0])/float(avail_freq[freq_indx]));  /* (1) Execution time prediction */
#endif
                            assembly->set_timetable(freq_indx, 0, ptt_value_newfreq, width-1);
                          }
                        }
                        for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* (2) Power value prediction (now memory-boundness is 0, cluster is Denver and width is known) */
                          // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
                            assembly->set_powertable(freq_indx, 0, runtime_power[0][freq_indx][0][width-1], width-1); /*Power: first parameter 0 is memory-boundness index*/
                          // }
                        }
                      }else{
                        for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* (1) Execution time prediction */
                          if(freq_indx == NUM_AVAIL_FREQ/2){
                            continue;
                          }else{
#if defined Performance_Model_Cycle 
                            uint64_t new_cycles = cycles_high * (1 - memory_boundness + memory_boundness * float(avail_freq[freq_indx])/float(avail_freq[0]));
                            float ptt_value_newfreq = float(new_cycles) / float(avail_freq[freq_indx]*1000);
#endif
#if defined Performance_Model_Time
                            float ptt_value_newfreq = highest_ticks * (memory_boundness + (1-memory_boundness) * float(avail_freq[0]) / float(avail_freq[freq_indx]));
#endif
                            assembly->set_timetable(freq_indx, 0, ptt_value_newfreq, width-1);
                          }
                        }
                        int mb_bound = floor(memory_boundness/0.1);  /* (2) Power value prediction */
                        for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                          assembly->set_powertable(freq_indx, 0, runtime_power[mb_bound][freq_indx][0][width-1], width-1); 
                        }
                      }
                    }
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
#if defined Performance_Model_Cycle    
                  std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.11GHz, Denver, " << width << "] = " << check_ticks << ". Cycles(1.11GHz) = " << check_cycles << ", Cycles(2.04GHz) = " << cycles_high <<\
                  ". Memory-boundness(Denver, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
#endif
#if defined Performance_Model_Time
                  std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[2.04GHz, Denver, " << width << "] = " << highest_ticks << ", PTT_Value[1.11GHz, Denver, " << width << "] = " << check_ticks << ". Memory-boundness(Denver, width=" << width << ") = " << memory_boundness \
                  << ". ptt_check = " << ptt_check << ".\n";
#endif
                  LOCK_RELEASE(output_lck);
#endif
                  }
                  continue;
                }else{
                  break;
                }
              }
              if(ptt_check == num_width[0]){ /* PTT at 1.11GHz are filled out */
                PTT_finish_state[1][0][assembly->tasktype] = 1;
                assembly->set_timetable_state(0, true); /* Finish the PTT training in Denver part, set state to true */
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] " << assembly->kernel_name << ": Denver cluster completed PTT training at 2.04GHz and 1.11GHz. \n";
                LOCK_RELEASE(output_lck);
#endif
              }
            }
          }else{ 
            mtx.lock();
            _final = (++assembly->threads_out_tao == assembly->width);
            mtx.unlock();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "Task " << assembly->taskid << "->_final = " << _final << ", assembly->get_timetable_state(Denver) = " << assembly->get_timetable_state(0) << ", assembly->start_running_freq = " \
            << assembly->start_running_freq<< ", cur_freq[Denver] = " << cur_freq[0] << "\n";
            LOCK_RELEASE(output_lck);
#endif        
          }    
        }

        /* (1) Was running on A57 (2) A57 PTT table hasn't finished training      (3) if the frequency changing is happening during the task execution, do not update the table */
        if(assembly->leader >= START_A){
          if(assembly->get_timetable_state(1) == false && assembly->start_running_freq == cur_freq[1]){ 
          if(ptt_freq_index[1] == 0){ /*2.04GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, 1, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || newticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 1, newticks, width_index);
                }
              }else{
                if(oldticks == 0.0f || ticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 1, ticks, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(0, 1, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[2.04GHz, A57, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(0,1, width_index) << ". Current time: " << assembly->get_timetable(0, 1, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
              //++assembly->threads_out_tao;
            }
//             float oldticks = assembly->get_timetable(0, 1, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){
//               assembly->set_timetable(0, 1, ticks, width_index); 
// #if defined Performance_Model_Cycle              
//               assembly->set_cycletable(0, 1, val1, width_index);
// #endif             
//             }
//             // else{
//             //   assembly->set_timetable(0, 1, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             // assembly->PTT_UpdateFinish[0][1][width_index]++;
//             assembly->increment_PTT_UpdateFinish(0, 1, width_index);
//             // }
          }else{ /*1.11GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 1, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || newticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(NUM_AVAIL_FREQ/2, 1, newticks, width_index);
                }
              }else{
                if(oldticks == 0.0f || ticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(NUM_AVAIL_FREQ/2, 1, ticks, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(1, 1, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[1.11GHz, A57, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(1,1, width_index) << ". Current time: " << assembly->get_timetable(NUM_AVAIL_FREQ/2, 1, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
//             float oldticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 1, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){
//               assembly->set_timetable(NUM_AVAIL_FREQ/2, 1, ticks, width_index); 
// #if defined Performance_Model_Cycle                 
//               assembly->set_cycletable(1, 1, val1, width_index);
// #endif              
//             }
//             // else{
//             //   assembly->set_timetable(NUM_AVAIL_FREQ/2, 1, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             // assembly->PTT_UpdateFinish[1][1][width_index]++;
//             assembly->increment_PTT_UpdateFinish(1, 1, width_index);
//             // }
          }
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_UpdateFinish[" << ptt_freq_index[1] << ", A57, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(ptt_freq_index[1], 1, width_index) << ".\n";
//             LOCK_RELEASE(output_lck);
// #endif
          // float oldticks = assembly->get_timetable(ptt_freq_index[1], 1, width_index);
          // if(oldticks == 0.0f){
          //   assembly->set_timetable(ptt_freq_index[1], 1, ticks, width_index); 
          // }else{
          //   assembly->set_timetable(ptt_freq_index[1], 1, ((oldticks + ticks)/2), width_index);  
          // }
          // //if(nthread == assembly->leader){
          //   PTT_UpdateFinish[ptt_freq_index[1]][1][width_index]++;

          //}
          /* Step 2: Update cycle table values */
          // uint64_t oldcycles = assembly->get_cycletable(ptt_freq_index[1], 1, width_index);
          // if(oldcycles == 0){
          //   assembly->set_cycletable(ptt_freq_index[1], 1, val1, width_index);
          // }else{
          //   assembly->set_cycletable(ptt_freq_index[1], 1, ((oldcycles + val1)/2), width_index);
          // }
          // if(assembly->get_timetable_state(1) == false){ /* PTT training hasn't finished */

          // if(nthread == assembly->leader){
            if (ptt_freq_index[1] == 0) {             /* Current A57 frequency is 2.04GHz */
              if(assembly->get_PTT_UpdateFinish(0, 1, 0) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 1, 1) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 1, 3) >= NUM_TRAIN_TASKS){ /* First dimention: 2.04GHz, second dimention: Denver, third dimention: width_index */
                PTT_finish_state[0][1][assembly->tasktype] = 1; /* First dimention: 2.04GHz, second dimention: Denver, third dimention: tasktype */
                if(std::accumulate(std::begin(PTT_finish_state[0][1]), std::begin(PTT_finish_state[0][1])+num_kernels, 0) == num_kernels){ /* Check all kernels have finished the training of Denver at 2.04GHz */
                  ARM << std::to_string(1113600) << std::endl;
                  ptt_freq_index[1] = 1;
                  cur_freq[1] = 1113600;
                  cur_freq_index[1] = NUM_AVAIL_FREQ/2;
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] A57 cluster completed all kernels' PTT training at 2.04GHz, turn to 1.11GHz. \n";
                  LOCK_RELEASE(output_lck);
#endif
                }
              }
//               int ptt_check = 0; /* Step 2: Check if PTT values of A57 are filled out */
//               for(auto&& width : ptt_layout[START_A]) { 
//                 float check_ticks = assembly->get_timetable(0, 1, width_index);
// #if defined Performance_Model_Cycle
//                 uint64_t check_cycles = assembly->get_cycletable(0, 1, width_index); 
//                 if(assembly->get_PTT_UpdateFinish(0, 1, width-1) >=NUM_TRAIN_TASKS && check_ticks > 0.0f && check_cycles > 0)
// #endif
// #if defined Performance_Model_Time
//                 if(assembly->get_PTT_UpdateFinish(0, 1, width-1) >=NUM_TRAIN_TASKS && check_ticks > 0.0f > 0)
// #endif
//                 {
//                   ptt_check++;
// #ifdef DEBUG
//                   LOCK_ACQUIRE(output_lck);
//                   std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[2.04GHz, A57, " << width << "] = " << check_ticks << ". ptt_check = " << ptt_check << ".\n";
//                   LOCK_RELEASE(output_lck);
// #endif
//                   continue;
//                 }else{
//                   break;
//                 }
//               }
//               if(ptt_check == num_width[1]){ /* All ptt values of A57 are positive => visited at least once, then configure the frequency to 1.11GHz */
//                 ptt_freq_index[1] = 1;
//                 ARM << std::to_string(1113600) << std::endl;
//                 cur_freq[1] = 1113600;
//                 cur_freq_index[1] = NUM_AVAIL_FREQ/2;
// #ifdef DEBUG
//                 LOCK_ACQUIRE(output_lck);
//                 std::cout << "[DEBUG] A57 cluster completed PTT training at 2.04GHz, turn to 1.11GHz. \n";
//                 LOCK_RELEASE(output_lck);
// #endif
//               }
            }else{
              int ptt_check = 0; /* Step 2: Check if PTT values of A57 are filled out */
              for(auto&& width : ptt_layout[START_A]) { 
                float check_ticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 1, width - 1);
                if(check_ticks > 0.0f && assembly->get_PTT_UpdateFinish(1, 1, width-1) >=NUM_TRAIN_TASKS){
                  ptt_check++;
                  if(assembly->get_mbtable(1, width-1) == 0.0f){
                    float memory_boundness = 0.0f;
#if defined Performance_Model_Cycle                  /* Calculate Memory-boundness = (1 - cycle2/cycle1) / (1- f2/f1) */
                    uint64_t check_cycles = assembly->get_cycletable(1, 1, width - 1);
                    uint64_t cycles_high = assembly->get_cycletable(0, 1, width-1); /* First parameter is 0, means 2.04GHz, check_cycles is at 1.11Ghz */
                    float a = 1 - float(check_cycles) / float(cycles_high);
                    float b = 1 - float(avail_freq[NUM_AVAIL_FREQ/2]) / float(avail_freq[0]);
                    memory_boundness = a/b;
#endif
#if defined Performance_Model_Time
                    float highest_ticks = assembly->get_timetable(0, 1, width - 1);
                    float a = float(avail_freq[0]) / float(avail_freq[NUM_AVAIL_FREQ/2]);
                    float b = check_ticks / highest_ticks;
                    memory_boundness = (b-a) / (1-a);
                    LOCK_ACQUIRE(output_lck);
                    std::cout << assembly->kernel_name << ": Memory-boundness Calculation (A57, width " << width << ") = " << memory_boundness << ". a = " << a << ", b = " << b << std::endl;
                    LOCK_RELEASE(output_lck);
#endif
                    assembly->set_mbtable(1, memory_boundness, width-1); /*first parameter: cluster 0 - Denver, second parameter: update value, third value: width_index */
                    if(memory_boundness > 1){
                      LOCK_ACQUIRE(output_lck);
                      std::cout << "[Warning] Memory-boundness Calculation (A57) is greater than 1!" << std::endl;
                      LOCK_RELEASE(output_lck);
                      memory_boundness = 1;
                      for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
#if defined Performance_Model_Cycle 
                        uint64_t new_cycles = cycles_high * float(avail_freq[freq_indx])/float(avail_freq[0]);
                        float ptt_value_newfreq = float(new_cycles) / float(avail_freq[freq_indx]*1000);
#endif
#if defined Performance_Model_Time
                        float ptt_value_newfreq = highest_ticks;
#endif
                        assembly->set_timetable(freq_indx, 1, ptt_value_newfreq, width-1);
                      }
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* (2) Power value prediction */
                          assembly->set_powertable(freq_indx, 1, runtime_power[9][freq_indx][1][width-1], width-1); 
                      }
                      // return -1;
                    }else{
                      if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
                        memory_boundness = 0.0001;
                        for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* Fill out PTT[other freqs]. Start from 1, which is next freq after 2.04, also skip 1.11GHz*/
                          if(freq_indx == NUM_AVAIL_FREQ/2){
                            continue;
                          }else{
#if defined Performance_Model_Cycle 
                            float ptt_value_newfreq = float(cycles_high) / float(avail_freq[freq_indx]*1000); /* (1) Execution time prediction */
#endif
#if defined Performance_Model_Time
                            float ptt_value_newfreq = highest_ticks * (float(avail_freq[0])/float(avail_freq[freq_indx]));  /* (1) Execution time prediction */
#endif
                            assembly->set_timetable(freq_indx, 1, ptt_value_newfreq, width-1);
                          }
                        }
                        for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* (2) Power value prediction (now memory-boundness is 0, cluster is A57 and width is known) */
                          // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
                            assembly->set_powertable(freq_indx, 1, runtime_power[0][freq_indx][1][width-1], width-1); /*Power: first parameter 0 is memory-boundness index*/
                          // }
                        }
                      }else{
                        for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                          if(freq_indx == NUM_AVAIL_FREQ/2){
                            continue;
                          }else{
#if defined Performance_Model_Cycle 
                            uint64_t new_cycles = cycles_high * (1 - memory_boundness + memory_boundness * float(avail_freq[freq_indx])/float(avail_freq[0]));
                            float ptt_value_newfreq = float(new_cycles) / float(avail_freq[freq_indx]*1000);
#endif
#if defined Performance_Model_Time
                            float ptt_value_newfreq = highest_ticks * (memory_boundness + (1-memory_boundness) * float(avail_freq[0]) / float(avail_freq[freq_indx]));
#endif                         
                            assembly->set_timetable(freq_indx, 1, ptt_value_newfreq, width-1);
                          }
                        }
                        int mb_bound = floor(memory_boundness/0.1);  /* (2) Power value prediction */
                        for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                          assembly->set_powertable(freq_indx, 1, runtime_power[mb_bound][freq_indx][1][width-1], width-1); 
                        }
                      }
                    }
#if (defined DEBUG)
                    LOCK_ACQUIRE(output_lck);
#if (defined Performance_Model_Cycle)
                    std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.11GHz, A57, " << width << "] = " << check_ticks << ". Cycles(1.11GHz) = " << check_cycles << ", Cycles(2.04GHz) = " << cycles_high <<\
                    ". Memory-boundness(A57, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
#endif
#if (defined Performance_Model_Time)
                    std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[2.04GHz, A57, " << width << "] = " << highest_ticks << ", PTT_Value[1.11GHz, A57, " << width << "] = " << check_ticks << ". Memory-boundness(A57, width=" << width << ") = " << memory_boundness \
                    << ". ptt_check = " << ptt_check << ".\n";
#endif
                    LOCK_RELEASE(output_lck);
#endif
                  }
                continue;
              }else{
                break;
              }
            }
            if(ptt_check == num_width[1]){ /* All ptt values of A57 are positive => visited at least once, then configure the frequency to 1.11GHz */
              PTT_finish_state[1][1][assembly->tasktype] = 1;
              assembly->set_timetable_state(1, true); /* Finish the PTT training in A57 part, set state to true */
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] " << assembly->kernel_name << ": A57 cluster completed PTT training at 2.04GHz and 1.11GHz. \n";
              LOCK_RELEASE(output_lck);
#endif
            }
          }
        // }
          }else{
            mtx.lock();
            _final = (++assembly->threads_out_tao == assembly->width);
            mtx.unlock();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "Task " << assembly->taskid << "->_final = " << _final << ", assembly->get_timetable_state(A57) = " << assembly->get_timetable_state(1) << ", assembly->start_running_freq = " \
            << assembly->start_running_freq<< ", cur_freq[A57] = " << cur_freq[1] << "\n";
            LOCK_RELEASE(output_lck);
#endif  
          }
        }

        if(assembly->get_timetable_state(2) == false && assembly->get_timetable_state(0) == true && assembly->get_timetable_state(1) == true){
          assembly->set_timetable_state(2, true);
          global_training_state[assembly->tasktype] = 1;
// #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << assembly->kernel_name << ": Training Phase finished. Predicted execution time and power results for the kernel tasks: \n";
          std::cout << "Execution Time Predictions: \n";
          for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
            std::cout << "Frequency: " << avail_freq[freq_indx] << ": " << std::endl;
            for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
              std::cout << "Cluster " << clus_id << ": ";
              for(int wid = 1; wid <= 4; wid *= 2){
                std::cout << assembly->get_timetable(freq_indx, clus_id, wid-1) << "\t";
              }
              std::cout << std::endl;
            }
          }
          std::cout << "\nPower Predictions: \n";
          for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
            std::cout << "Frequency: " << avail_freq[freq_indx] << ": " << std::endl;
            for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
              std::cout << "Cluster " << clus_id << ": ";
              for(int wid = 1; wid <= 4; wid *= 2){
                std::cout << assembly->get_powertable(freq_indx, clus_id, wid-1) << "\t";
              }
              std::cout << std::endl;
            }
          }
          std::cout << std::endl;
          LOCK_RELEASE(output_lck);
// #endif
        }
        if(std::accumulate(std::begin(global_training_state), std::begin(global_training_state) + num_kernels, 0) == num_kernels) {// Check if all kernels have finished the training phase.
          global_training = true;
          std::chrono::time_point<std::chrono::system_clock> train_end;
          train_end = std::chrono::system_clock::now();
          auto train_end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(train_end);
          auto train_end_epoch = train_end_ms.time_since_epoch();
          LOCK_ACQUIRE(output_lck);
          std::cout << "[Congratulations!] All the training Phase finished. Trainging finished time: " << train_end_epoch.count() << ". " << std::endl;
          LOCK_RELEASE(output_lck);
        }
#if 0
        if(assembly->leader == nthread){
          //std::atomic_fetch_sub(&status, 1);
          //std::chrono::duration<double> elapsed_seconds = t2-t1;
          double ticks = elapsed_seconds.count();
          int width_index = assembly->width - 1;
          //Weight the newly recorded ticks to the old ticks 1:4 and save
#if (defined TX2) && (defined DVFS)
          // Old DVFS code: PTT is defined for each frequency level - Not scalable 
          if(nthread < 2){
            float oldticks = assembly->get_timetable(denver_freq, nthread, width_index);
            if(oldticks == 0){
              assembly->set_timetable(denver_freq, nthread,ticks,width_index);  
            }
            else {
              assembly->set_timetable(denver_freq, nthread,((4*oldticks+ticks)/5),width_index);         
            }
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DVFS] Task " << assembly->taskid << ", 1) on Denver execution time: " << ticks << ", " << "assembly->get_timetable(" << denver_freq << "," << nthread << "," << width_index <<") = " << assembly->get_timetable(denver_freq, nthread,width_index) << "\n";
            LOCK_RELEASE(output_lck);
#endif 
          }else{
            float oldticks = assembly->get_timetable(a57_freq, nthread,width_index);
            if(oldticks == 0){
              assembly->set_timetable(a57_freq, nthread,ticks,width_index);  
            }
            else {
              assembly->set_timetable(a57_freq, nthread,((4*oldticks+ticks)/5),width_index);         
            }
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DVFS] Task " << assembly->taskid << ", 1) on A57 execution time: " << ticks << ", after updating, time is " << assembly->get_timetable( a57_freq, nthread,width_index) << "\n";
            LOCK_RELEASE(output_lck);
#endif 
          }
#elif (defined DynaDVFS) && (defined PERF_COUNTERS)
          // 2021 - 08 - 24 TACO Review of Experiemnts: ERASE Reaction to Dynamic DVFS Change 
          
          float oldticks = assembly->get_timetable(nthread,width_index);
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] oldticks = " << oldticks << ". Increase/decrease(%): " << abs(ticks - oldticks)/oldticks << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
          if(oldticks == 0){
            assembly->set_timetable(nthread,ticks,width_index);  
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Oldticks=0. New updated execution time = " << ticks << " by task " << assembly->taskid << std::endl;
            LOCK_RELEASE(output_lck);
#endif            
          }else{
            int computed_freq = val1 / (ticks*1000);
            float freq_deviation = fabs(1 - (float)computed_freq/(float)current_freq);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] computed freq = " << computed_freq << ". Real frequency = " << current_freq << ". Deviation = " << freq_deviation*100 << "%." << std::endl;
            LOCK_RELEASE(output_lck);
#endif
            // Frequency devistion < 3% => frequency is not changed
            if(freq_deviation < 0.03){
              assembly->set_timetable(nthread,((4*oldticks + ticks)/5),width_index);
            }else{
              std::ifstream freq_file;
              freq_file.open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
              if(freq_file.fail()){
                std::cout << "Failed to open sys frequency reading file!" << std::endl;
                std::cin.get();
                return 0;
              }
              int reading_freq = 0;
              while(freq_file >> reading_freq); // Read value and store in reading_freq
              freq_file.close();
              if(reading_freq != current_freq){ // Check again if the frequency changed or not
                current_freq = reading_freq; // changed, rewrite the current frequency
                for(int leader = 0; leader < ptt_layout.size(); ++leader) { // Reset PTT entries to zeros
                  for(auto&& width : ptt_layout[leader]) {
                    assembly->set_timetable(leader,0.0,width-1);
                  }
                }
                ptt_full = false;
                assembly->set_timetable(nthread,ticks,width_index);
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] Frequency change is confirmed. current frequency = " << current_freq << ". Reset PTT = 0. PTT_full = false. Task " << assembly->taskid << " update PTT(" \
                << nthread << "," << assembly->width << ") with new time " << ticks << std::endl;
                LOCK_RELEASE(output_lck);
#endif
              } 
            }

            // Scalable Alternative: Detect the new execution time, if the difference between the old and the new is 
            // more than 10% (confidence interval), consider the runtime is doing frequency change, then reset the value
            // in this cell. If detecting n (=5) cases, then we are sure there is a frequency change. Reset other values
            // to zero and retrain the PTT table.

//             // float rate = abs(float)(ticks - oldticks)/(float)oldticks;
//             float rate = fabs(1 - (float)ticks/(float)oldticks);
//             if(rate < 0.3){
//               assembly->set_timetable(nthread,((4*oldticks + ticks)/5),width_index);
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] |OldTime-NewTime|<30%. oldticks = " << oldticks << ". Increase/decrease(%): " << rate \
//               << ". Task " << assembly->taskid << " update time with weight(1/5). After updating: " << (4*oldticks + ticks)/5 << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif
//             }else{
//             //if((float)(abs(ticks - oldticks))/oldticks >= 0.15){
//               dynamic_time_change++;
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Consider frequency change: |OldTime-NewTime|>=30%. Oldticks: " << oldticks << ". Update with the new value from Task " \
//               << assembly->taskid << ": " << ticks << ". Dynamic_time_change = " << dynamic_time_change << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif
//               if (ticks > oldticks){
//                 freq_dec++; // Frequency is decreasing
//               }else{
//                 freq_inc++; // Frequency is increasing
//               }

//               if(dynamic_time_change > 4){
// #ifdef DEBUG
//                 LOCK_ACQUIRE(output_lck);
//                 std::cout << "[DEBUG] >4 tasks show frequency change. Reset all PTT entries to zeros." << std::endl;
//                 LOCK_RELEASE(output_lck);
// #endif
//                 // Reset PTT entries to zeros
//                 for(int leader = 0; leader < ptt_layout.size(); ++leader) {
//                   for(auto&& width : ptt_layout[leader]) {
//                     assembly->set_timetable(leader,0.0,width-1);
//                   }
//                 }
//                 ptt_full = false;
//                 dynamic_time_change = 0;
//               }
//               assembly->set_timetable(nthread,ticks,width_index);
//             }
          }
          
#else
          float oldticks = assembly->get_timetable( nthread,width_index);
          if(oldticks == 0){
            assembly->set_timetable(nthread,ticks,width_index);  
          }
          else {
#ifdef PTTaccuracy
            /* Final PTT accuracy = SUM up all MAE and divided by number of tasks then multiply 100% */
            if(assembly->finalpredtime != 0.0f){
              MAE += 1 - (fabs(ticks - assembly->finalpredtime) / ticks);
              PTT << ticks << "\t" << assembly->finalpredtime << "\n";
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] Task " << assembly-> taskid << " time prediction: " << assembly->finalpredtime << ", execution time: " << ticks << ", error: " << fabs(ticks - assembly->finalpredtime) << ", MAE = " << MAE << std::endl;
              LOCK_RELEASE(output_lck);
#endif
            }
#endif
            assembly->set_timetable(nthread,((oldticks + ticks)/2),width_index);         
          }
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] Task " << assembly-> taskid << ", execution time: " << ticks << ", after updating, time is " << assembly->get_timetable( nthread,width_index) << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif  
#endif
#ifdef Energyaccuracy
#if (defined TX2) && (defined MultipleKernels)
            if(assembly->finalenergypred == 0.0f){
              if(assembly->finalpowerpred != 0.0f){
                assembly->finalenergypred = assembly->finalpowerpred * ticks;
              }else{
                if(assembly->tasktype == 0){
                  assembly->finalpowerpred = (nthread < 2)? (compute_bound_power[0][assembly->width] +compute_bound_power[0][0]): (compute_bound_power[1][assembly->width]+compute_bound_power[1][0]);
                  assembly->finalenergypred = assembly->finalpowerpred * ticks;
                }else{
                  if(assembly->tasktype == 1){
                    assembly->finalpowerpred = (nthread < 2)? memory_bound_power[0][assembly->width] +memory_bound_power[0][0] : memory_bound_power[1][assembly->width] +memory_bound_power[1][0];
                    assembly->finalenergypred = assembly->finalpowerpred * ticks;
                  }
                }
              }
            }
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Task " << assembly->taskid <<" execution time = " << ticks << ", Predicted energy / power =  " << assembly->finalenergypred << ". \n";
            LOCK_RELEASE(output_lck);
#endif
#endif
#ifndef MultipleKernels
          if(ptt_full)
#endif
          {
            EnergyPrediction += assembly->finalenergypred;
          }
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << nthread << " energy pred sum =  " << EnergyPrediction << std::endl;
          LOCK_RELEASE(output_lck);
#endif  
#endif
          /* Write Energy into PTT Table */
          //assembly->set_timetable(nthread,ticks*assembly->get_power(nthread, assembly->width),width_index);
// #ifdef DEBUG
          // LOCK_ACQUIRE(output_lck);
//           //std::cout << assembly->get_timetable(0,0) << "," << assembly->get_timetable(0,1) << "," << assembly->get_timetable(1,0) << "," << assembly->get_timetable(2,0) << "," << assembly->get_timetable(2,1) << "," << assembly->get_timetable(2,3) << "," << assembly->get_timetable(3,0) << "," << assembly->get_timetable(4,0) << "," << assembly->get_timetable(4,1) << "," << assembly->get_timetable(5,0) << "\n";
//           //std::cout << "[CHEN] Task " << assembly-> taskid << ", 1) execution time: " << ticks << ", after updating, time is " << assembly->get_timetable( nthread,width_index) << ", 2) Power: " << assembly->get_power(nthread, width_index+1) << ", 3) Energy: " << assembly->get_timetable( nthread,width_index)* assembly->get_power(nthread, width_index+1) << std::endl;
//           std::cout << "[DEBUG] Task " << assembly-> taskid << ", 1) execution time: " << ticks << ", after updating, time is " << assembly->get_timetable( nthread,width_index) << std::endl;
//           //", 2) Power: " << assembly->get_power(nthread, width_index+1) <<  ", 3) Energy[" << nthread <<"]: " << energy[nthread] << std::endl;
          // LOCK_RELEASE(output_lck);
// #endif   
        }
#endif 
      }
      else{ /* Other No training tasks or Other kinds of schedulers*/
        mtx.lock();
        _final = (++assembly->threads_out_tao == assembly->width);
        mtx.unlock();
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Task " << assembly->taskid << ", _final = " << _final << "\n";
        LOCK_RELEASE(output_lck);
#endif        
      }
    st = nullptr;
    if(_final){ // the last exiting thread updates
      task_completions[nthread].tasks++;
      if(task_completions[nthread].tasks > 0){
        PolyTask::pending_tasks -= task_completions[nthread].tasks;
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Thread " << nthread << " completed " << task_completions[nthread].tasks << " task " << assembly->taskid <<". Pending tasks = " << PolyTask::pending_tasks << "\n";
        LOCK_RELEASE(output_lck);
#endif
        task_completions[nthread].tasks = 0;
      }
      
#ifdef OVERHEAD_PTT
      st = assembly->commit_and_wakeup(nthread, elapsed_ptt);
#else
      st = assembly->commit_and_wakeup(nthread);
#endif
      assembly->cleanup();
    }
    idle_try = 0;
    idle_times = 0;
    continue;
  }

    // 2. check own queue
    if(!stop){
      LOCK_ACQUIRE(worker_lock[nthread]);
      if(!worker_ready_q[nthread].empty()){
        st = worker_ready_q[nthread].front(); 
        worker_ready_q[nthread].pop_front();
        LOCK_RELEASE(worker_lock[nthread]);
        continue;
      }     
      LOCK_RELEASE(worker_lock[nthread]);        
    }

#ifdef WORK_STEALING
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[Test] Thread " << nthread << " goes out of work stealing.\n";
//       LOCK_RELEASE(output_lck);          
// #endif

    // 3. try to steal rand_r(&seed)
// #ifdef ERASE_target_edp
//     if((rand() % A57_best_edp_width == 0) && !stop)
// #else
    if((rand() % STEAL_ATTEMPTS == 0) && !stop)
// #endif
    {
      status_working[nthread] = 0;
      int attempts = gotao_nthreads;
#ifdef SLEEP
#if (defined RWSS_SLEEP)
      if(Sched == 3){
        idle_try++;
      }

#endif
#if (defined FCAS_SLEEP)
      if(Sched == 0){
        idle_try++;
      }
#endif
#if (defined EAS_SLEEP)
      if(Sched == 1){
        idle_try++;
      }
#endif
#endif

      do{
        if(Sched == 2){
          if(DtoA <= maySteal_DtoA){
            do{
              random_core = (rand_r(&seed) % gotao_nthreads);
            } while(random_core == nthread);
           
          }
          else{
            if(nthread < 2){
              do{
                random_core = (rand_r(&seed) % 2);
              } while(random_core == nthread);
            }else{
         	    do{
                random_core = 2 + (rand_r(&seed) % 4);
              }while(random_core == nthread); 
            }
          }
        }
        //EAS
        if(Sched == 1){
#if defined(TX2)
// #ifndef MultipleKernels
//         	if(!ptt_full){
          	// do{
            // 	random_core = (rand_r(&seed) % gotao_nthreads);
          	// } while(random_core == nthread);
//         	}else
// #endif
          // {
// #if (defined ERASE_target_perf) || (defined ERASE_target_edp_method1)
//           if(D_give_A == 0 || steal_DtoA < D_give_A){
//             int Denver_workload = worker_ready_q[0].size() + worker_ready_q[1].size();
//             int A57_workload = worker_ready_q[2].size() + worker_ready_q[3].size() + worker_ready_q[4].size() + worker_ready_q[5].size();
//           // If there is more workload to share with A57 
//             D_give_A = (Denver_workload-A57_workload) > 0? floor((Denver_workload-A57_workload) * 1 / (D_A+1)) : 0; 
//             //if((Denver_workload-A57_workload) > 0 && D_give_A > 0){
//             if(D_give_A > 0){
//               random_core = rand_r(&seed) % 2;
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] There is more workload that can be shared with A57. Size =  " << D_give_A << ". \n";
//               LOCK_RELEASE(output_lck);          
// #endif
//             }
//           }else
// #endif
          // { 
          	/* [EAS] Only steal tasks from same cluster */
            if(nthread < 2){
            	do{
              	random_core = (rand_r(&seed) % (START_A - START_D));
            	} while(random_core == nthread);
          	}else{
         	  	do{
              	random_core = START_A + (rand_r(&seed) % (gotao_nthreads - START_A));
            	}while(random_core == nthread); 
          	}

					// }    
        // }
#endif
#if defined(Haswell)
          // if(nthread < gotao_nthreads/NUMSOCKETS){
          //   do{
          //     random_core = (rand_r(&seed) % (gotao_nthreads/NUMSOCKETS));
          //   } while(random_core == nthread);
          // }else{
          //   do{
          //     random_core = (gotao_nthreads/NUMSOCKETS) + (rand_r(&seed) % (gotao_nthreads/NUMSOCKETS));
          //   } while(random_core == nthread);
          // }
          do{
            random_core = (rand_r(&seed) % gotao_nthreads);
          } while(random_core == nthread);
#endif
        }

        if(Sched == 0){
#ifdef CATS
          if(nthread > 1){
            do{
              random_core = 2 + (rand_r(&seed) % 4);
            } while(random_core == nthread);
          }else{
            do{
              random_core = (rand_r(&seed) % gotao_nthreads);
            } while(random_core == nthread);
          }
#else
				  do{
            random_core = (rand_r(&seed) % gotao_nthreads);
          } while(random_core == nthread);
#endif
        }

        if(Sched == 3){
				  do{
            random_core = (rand_r(&seed) % gotao_nthreads);
          } while(random_core == nthread);
				}

        LOCK_ACQUIRE(worker_lock[random_core]);
        if(!worker_ready_q[random_core].empty()){
          st = worker_ready_q[random_core].back();
					if((Sched == 1) || (Sched == 2)){
            // [EAS] Not steal tasks from same pair, e.g, thread 0 does not steal the task width=2 from thread 1.
            // [EDP] Not steal tasks when ready queue task size is only 1.
            if((st->width >= abs(random_core-nthread)+1)) {
              st = NULL;
              LOCK_RELEASE(worker_lock[random_core]);
              continue;
            }
            else{
              // if(Sched == 2 && DtoA <= maySteal_DtoA){
              //   if(random_core > 1 && nthread < 2){
              //     std::atomic_fetch_sub(&DtoA, 1);
              //   }
              //   else{
              //     if(random_core < 2 && nthread > 1){
              //       if(worker_ready_q[random_core].size() <= 4){
              //         st = NULL;
              //         LOCK_RELEASE(worker_lock[random_core]);
              //         continue;
              //       }
              //       std::atomic_fetch_add(&DtoA, 1);
              //     }
              //   }
              //   //std::cout << "Steal D to A is " << DtoA << "\n";
              // }
              worker_ready_q[random_core].pop_back();
              
// #if (defined ERASE_target_edp_method1)
//               if(ptt_full==true && nthread > 1 && random_core < 2){
//                 if((steal_DtoA++) == D_give_A){
//                   steal_DtoA = 0;
//                 }
//                 st->width = A57_best_edp_width;
//                 if(st->width == 4){
//                   st->leader = 2 + (nthread-2) / st->width;
//                 }
//                 if(st->width <= 2){
//                   st->leader = nthread /st->width * st->width;
//                 }
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Full PTT: Thread " << nthread << " steal task " << st->taskid << " from " << random_core << " successfully. Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
//               LOCK_RELEASE(output_lck);          
// #endif	
//               }else{
//                 st->leader = nthread /st->width * st->width;
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Other: Thread " << nthread << " steal task " << st->taskid << " from " << random_core << " successfully. Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
//               LOCK_RELEASE(output_lck);          
// #endif	
//               }       
      
// #endif

#if (defined ERASE_target_perf) 
#ifndef MultipleKernels
              // if(ptt_full == true){
                st->history_mold(nthread, st); 
              // }
              // else{
              //   st->eas_width_mold(nthread, st);    
              // }
#endif             
#endif
              if(st->width == 4){
                st->leader = 2 + (nthread-2) / st->width;
              }
              if(st->width <= 2){
                st->leader = nthread /st->width * st->width;
              }
           
              // if(st->get_bestconfig_state() == true && nthread >= 2){ //Test Code: Allow work stealing across clusters
              //   st->width = 4;
              //   st->leader = 2;
              // }
              tao_total_steals++;  
            }
// #ifdef Energyaccuracy
//             if(st->finalenergypred == 0.0f){
//               st->finalenergypred = st->finalpowerpred * st->get_timetable(st->leader, st->width - 1);
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Task " << st->taskid << " prediction energy = " << st->finalenergypred << ". \n";
//               LOCK_RELEASE(output_lck);          
// #endif
//             }
// #endif
          }
          else{
            if((Sched == 0) || (Sched == 3)){
            // if((st->criticality == 0 && Sched == 0) ){
              worker_ready_q[random_core].pop_back();
              st->leader = nthread /st->width * st->width;
              tao_total_steals++;
            }else{
              st = NULL;
              LOCK_RELEASE(worker_lock[random_core]);
              continue;
            }
          }

#ifndef CATS				
          if(Sched == 0){
            st->history_mold(nthread, st);    
#ifdef DEBUG
          	LOCK_ACQUIRE(output_lck);
						std::cout << "[DEBUG] Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
						LOCK_RELEASE(output_lck);
#endif
          }
#endif
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << nthread << " steal task " << st->taskid << " from " << random_core << " successfully. Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
          LOCK_RELEASE(output_lck);          
#endif	
//           if(Sched == 1){
//             st->eas_width_mold(nthread, st);    
// #ifdef DEBUG
//           	LOCK_ACQUIRE(output_lck);
// 						std::cout << "[EAS-STEAL] Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
// 						LOCK_RELEASE(output_lck);
// #endif
//           }
          
        }
        LOCK_RELEASE(worker_lock[random_core]);  
      }while(!st && (attempts-- > 0));
      if(st){
#ifdef SLEEP
#if (defined RWSS_SLEEP)
        if(Sched == 3){
          idle_try = 0;
          idle_times = 0;
        }
#endif
#if (defined FCAS_SLEEP)
        if(Sched == 0){
          idle_try = 0;
          idle_times = 0;
        }
#endif
#if (defined EAS_SLEEP)
        if(Sched == 1){
          idle_try = 0;
          idle_times = 0;
        }
#endif
#endif
        status_working[nthread] = 1;
        continue;
      }
    }
#endif
    
/*
    if(Sched == 1){
      if(idle_try >= idle_sleep){
        if(!stop){
			  idle_times++;
        usleep( 100000 * idle_times);
        if(idle_times >= forever_sleep){
          // Step1: disable PTT entries of the thread
          for (int j = 0; j < inclusive_partitions[nthread].size(); ++j){
					  PTT_flag[inclusive_partitions[nthread][j].second - 1][inclusive_partitions[nthread][j].first] = 0;
					  //std::cout << "PTT_flag[" << inclusive_partitions[nthread][j].second - 1 << "]["<<inclusive_partitions[nthread][j].first << "] = 0.\n";
    		  }
          // Step 2: Go back to main work loop to check AQ 
          stop = true;
          continue;
        }
        }
      }
      if(stop){
        // Step 3: Go to sleep forever 
        std::cout << "Thread " << nthread << " go to sleep forever!\n";
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, []{return finish;});
        lk.unlock();
        break;
      }
    }
*/
#if (defined SLEEP) 
      if(idle_try >= IDLE_SLEEP){
        long int limit = (SLEEP_LOWERBOUND * pow(2,idle_times) < SLEEP_UPPERBOUND) ? SLEEP_LOWERBOUND * pow(2,idle_times) : SLEEP_UPPERBOUND;  
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);      
//         std::cout << "Thread " << nthread << " sleep for " << limit/1000000 << " ms.\n";
//         LOCK_RELEASE(output_lck);
// #endif
        status[nthread] = 0;
        status_working[nthread] = 0;
        tim.tv_sec = 0;
        tim.tv_nsec = limit;
        nanosleep(&tim , &tim2);
        //SleepNum++;
        AccumTime += limit/1000000;
        idle_times++;
        idle_try = 0;
        status[nthread] = 1;
      }
#endif
    // 4. It may be that there are no more tasks in the flow
    // this condition signals termination of the program
    // First check the number of actual tasks that have completed
//     if(task_completions[nthread].tasks > 0){
//       PolyTask::pending_tasks -= task_completions[nthread].tasks;
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] Thread " << nthread << " completed " << task_completions[nthread].tasks << " tasks. Pending tasks = " << PolyTask::pending_tasks << "\n";
//       LOCK_RELEASE(output_lck);
// #endif
//       task_completions[nthread].tasks = 0;
//     }
    LOCK_ACQUIRE(worker_lock[nthread]);
    // Next remove any virtual tasks from the per-thread task pool
    if(task_pool[nthread].tasks > 0){
      PolyTask::pending_tasks -= task_pool[nthread].tasks;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] Thread " << nthread << " removed " << task_pool[nthread].tasks << " virtual tasks. Pending tasks = " << PolyTask::pending_tasks << "\n";
      LOCK_RELEASE(output_lck);
#endif
      task_pool[nthread].tasks = 0;
    }
    LOCK_RELEASE(worker_lock[nthread]);
    
    // Finally check if the program has terminated
    if(gotao_can_exit && (PolyTask::pending_tasks == 0)){
#ifdef PowerProfiling
      out.close();
#endif
      
#ifdef SLEEP
      LOCK_ACQUIRE(output_lck);
      std::cout << "Thread " << nthread << " sleeps for " << AccumTime << " ms. \n";
      LOCK_RELEASE(output_lck);
#endif

#ifdef PTTaccuracy
      LOCK_ACQUIRE(output_lck);
      std::cout << "Thread " << nthread << " 's MAE = " << MAE << ". \n";
      LOCK_RELEASE(output_lck);
      PTT.close();
#endif

#ifdef Energyaccuracy
      LOCK_ACQUIRE(output_lck);
      std::cout << "Thread " << nthread << " 's Energy Prediction = " << EnergyPrediction << ". \n";
      LOCK_RELEASE(output_lck);
#endif

#ifdef NUMTASKS_MIX
      LOCK_ACQUIRE(output_lck);
#ifdef TX2
      for(int b = 0;b < XITAO_MAXTHREADS; b++){
        for(int a = 1; a < gotao_nthreads; a = a*2){
          std::cout << "Task type: " << b << ": Thread " << nthread << " with width " << a << " completes " << num_task[b][a * gotao_nthreads + nthread] << " tasks.\n";
          num_task[b][a * gotao_nthreads + nthread] = 0;
        }
      }
      // NUM_WIDTH_TASK[1] += num_task[1 * gotao_nthreads + nthread];
      // NUM_WIDTH_TASK[2] += num_task[2 * gotao_nthreads + nthread];
      // NUM_WIDTH_TASK[4] += num_task[4 * gotao_nthreads + nthread];
      // num_task[1 * gotao_nthreads + nthread] = 0;
      // num_task[2 * gotao_nthreads + nthread] = 0;
      // num_task[4 * gotao_nthreads + nthread] = 0;
#endif
      LOCK_RELEASE(output_lck);
#endif
#ifdef EXECTIME
      LOCK_ACQUIRE(output_lck);
      std::cout << "The total execution time of thread " << nthread << " is " << elapsed_exe.count() << " s.\n";
      LOCK_RELEASE(output_lck);
#endif
#ifdef OVERHEAD_PTT
      LOCK_ACQUIRE(output_lck);
      std::cout << "PTT overhead of thread " << nthread << " is " << elapsed_ptt.count() << " s.\n";
      LOCK_RELEASE(output_lck);
#endif
      // if(Sched == 0){
      //break;
      // }else{
      return 0;
      // }
    }
  }
  // pmc.close();
  Denver.close();
  ARM.close();
  return 0;
}
