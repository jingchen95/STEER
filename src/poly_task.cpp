#include "poly_task.h"
// #include "tao.h"
#include <errno.h> 
#include <cstring>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iterator>
#include <chrono>
#include <cmath>
#include <fstream>
#include "xitao_workspace.h"
using namespace xitao;

const int EAS_PTT_TRAIN = 0;

extern int freq_index;
extern long cur_freq[NUMSOCKETS];
extern int cur_freq_index[NUMSOCKETS];
extern long avail_freq[NUM_AVAIL_FREQ];
extern int num_width[NUMSOCKETS];
extern int ptt_freq_index[NUMSOCKETS];
extern bool global_training;
extern int num_kernels;

#if defined(TX2)
int idleP_A57 = 152;
int idleP_Denver = 76;
#endif

#if (defined DynaDVFS)
// extern int freq_dec;
// extern int freq_inc;
extern int env;
extern int current_freq;
// extern int dynamic_time_change;
#endif

extern int denver_freq;
extern int a57_freq;

#if defined(Haswell)
extern int num_sockets;
#endif

#ifdef OVERHEAD_PTT
extern std::chrono::duration<double> elapsed_ptt;
#endif

#ifdef NUMTASKS_MIX
extern int num_task[XITAO_MAXTHREADS][XITAO_MAXTHREADS * XITAO_MAXTHREADS];
#endif

// #ifdef DVFS
// extern int PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
// #else
// extern int PTT_UpdateFlag[XITAO_MAXTHREADS][XITAO_MAXTHREADS];
// #endif
extern int start_coreid[NUMSOCKETS];
extern int end_coreid[NUMSOCKETS];

// std::ofstream Denver("/sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
// std::ofstream ARM("/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");

extern std::ofstream Denver;
extern std::ofstream ARM;

#if (defined DVFS) && (defined TX2)
extern float compute_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS];
extern float memory_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS];
extern float cache_intensive_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS];
#elif (defined DynaDVFS) // Currently only consider 4 combinations: max&max, max&min, min&max, min&min
extern float compute_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS];
extern float memory_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS];
extern float cache_intensive_power[4][NUMSOCKETS][XITAO_MAXTHREADS];
#elif (defined ERASE)
extern float compute_bound_power[NUMSOCKETS][XITAO_MAXTHREADS];
extern float memory_bound_power[NUMSOCKETS][XITAO_MAXTHREADS];
extern float cache_intensive_power[NUMSOCKETS][XITAO_MAXTHREADS];
#else
extern float runtime_power[10][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
extern float idle_power[NUM_AVAIL_FREQ][NUMSOCKETS];
#endif


extern int status[XITAO_MAXTHREADS];
extern int status_working[XITAO_MAXTHREADS];
extern int Parallelism;
extern int Sched;
extern int TABLEWIDTH;

//The pending PolyTasks count 
std::atomic<int> PolyTask::pending_tasks;

// need to declare the static class memeber
#if defined(DEBUG)
std::atomic<int> PolyTask::created_tasks;
#endif

PolyTask::PolyTask(int t, int _nthread=0) : type(t){
  refcount = 0;
#define GOTAO_NO_AFFINITY (1.0)
  affinity_relative_index = GOTAO_NO_AFFINITY;
  affinity_queue = -1;
#if defined(DEBUG) 
  taskid = created_tasks += 1;
#endif
  LOCK_ACQUIRE(worker_lock[_nthread]);
  if(task_pool[_nthread].tasks == 0){
    pending_tasks += TASK_POOL;
    task_pool[_nthread].tasks = TASK_POOL-1;
#ifdef DEBUG
    std::cout << "[DEBUG] _nthread = " << _nthread << ". Requested: " << TASK_POOL << " tasks. Pending is now: " << pending_tasks << "\n";
#endif
  }
  else {
    task_pool[_nthread].tasks--;
// #ifdef DEBUG
//     std::cout << "[Jing] task_pool[" << _nthread << "]--: " << task_pool[_nthread].tasks << "\n";
// #endif
  }
  LOCK_RELEASE(worker_lock[_nthread]);
/*	LOCK_ACQUIRE(worker_lock[0]);
  if(task_pool[0].tasks == 0){
    pending_tasks += TASK_POOL;
    task_pool[0].tasks = TASK_POOL-1;
#ifdef DEBUG
    std::cout << "[DEBUG] Requested: " << TASK_POOL << " tasks. Pending is now: " << pending_tasks << "\n";
#endif
  }
  else {
    task_pool[0].tasks--;
#ifdef DEBUG
    std::cout << "[Jing] task_pool[0]--: " << task_pool[0].tasks << "\n";
#endif
  }
  LOCK_RELEASE(worker_lock[0]);*/
  threads_out_tao = 0;
  criticality=0;
  marker = 0;
}

// Internally, GOTAO works only with queues, not stas
int PolyTask::sta_to_queue(float x){
  if(x >= GOTAO_NO_AFFINITY){ 
    affinity_queue = -1;
  }
    else if (x < 0.0) return 1;  // error, should it be reported?
    else affinity_queue = (int) (x*gotao_nthreads);
    return 0; 
  }
int PolyTask::set_sta(float x){    
  affinity_relative_index = x;  // whenever a sta is changed, it triggers a translation
  return sta_to_queue(x);
} 
float PolyTask::get_sta(){             // return sta value
  return affinity_relative_index; 
}    
int PolyTask::clone_sta(PolyTask *pt) { 
  affinity_relative_index = pt->affinity_relative_index;    
  affinity_queue = pt->affinity_queue; // make sure to copy the exact queue
  return 0;
}
void PolyTask::make_edge(PolyTask *t){
  out.push_back(t);
  t->refcount++;
}

//History-based molding
//#if defined(CRIT_PERF_SCHED)
int PolyTask::history_mold(int _nthread, PolyTask *it){
  int new_width = 1; 
  int new_leader = -1;
  float shortest_exec = 1000.0f;
  float comp_perf = 0.0f; 
  auto&& partitions = inclusive_partitions[_nthread];
// #ifndef ERASE_target_perf
//   if(rand()%10 != 0) {
// #endif 
    for(auto&& elem : partitions) {
      int leader = elem.first;
      int width  = elem.second;
      auto&& ptt_val = 0.0f;
#ifdef DVFS
#else
      ptt_val = it->get_timetable(0, leader, width - 1);
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout <<"[DEBUG] Priority=0, ptt value("<< leader << "," << width << ") = " << ptt_val << std::endl;
      LOCK_RELEASE(output_lck);
#endif
#endif
      if(ptt_val == 0.0f) {
        new_width = width;
        new_leader = leader;       
        break;
      }
#ifdef CRI_COST
      //For Cost
      comp_perf = width * ptt_val;
#endif
#if (defined CRI_PERF) || (defined ERASE_target_perf) 
      //For Performance
      comp_perf = ptt_val;
#endif
      if (comp_perf < shortest_exec) {
        shortest_exec = comp_perf;
        new_width = width;
        new_leader = leader;      
      }
    }
// #ifndef ERASE_target_perf
//   } else { 
//     auto&& rand_partition = partitions[rand() % partitions.size()];
//     new_leader = rand_partition.first;
//     new_width  = rand_partition.second;
//   }
// #endif
  if(new_leader != -1) {
    it->width  = new_width;
    it->leader = new_leader;
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout <<"[DEBUG] History Mold, task "<< it->taskid << " leader: " << it->leader << ", width: " << it->width << std::endl;
    LOCK_RELEASE(output_lck);
#endif
  }
  return it->leader;
} 
  //Recursive function assigning criticality
int PolyTask::set_criticality(){
  if((criticality)==0){
    int max=0;
    for(std::list<PolyTask *>::iterator edges = out.begin();edges != out.end();++edges){
      int new_max =((*edges)->set_criticality());
      max = ((new_max>max) ? (new_max) : (max));
    }
    criticality=++max;
  } 
  return criticality;
}

int PolyTask::set_marker(int i){
  for(std::list<PolyTask *>::iterator edges = out.begin(); edges != out.end(); ++edges){
    if((*edges) -> criticality == critical_path - i){
      (*edges) -> marker = 1;
      i++;
      (*edges) -> set_marker(i);
      break;
    }
  }
  return marker;
}

//Determine if task is critical task
int PolyTask::if_prio(int _nthread, PolyTask * it){
#ifdef EAS_NoCriticality
	if((Sched == 1) || (Sched == 2)){
	  return 0;
  }
	if(Sched == 0){
#endif
    return it->criticality;
#ifdef EAS_NoCriticality
  }
#endif
}

// #ifdef DVFS
// void PolyTask::print_ptt(float table[][XITAO_MAXTHREADS][XITAO_MAXTHREADS], const char* table_name) { 
// #else
void PolyTask::print_ptt(float table[][XITAO_MAXTHREADS], const char* table_name) { 
// #endif
  std::cout << std::endl << table_name <<  " PTT Stats: " << std::endl;
  for(int leader = 0; leader < ptt_layout.size() && leader < gotao_nthreads; ++leader) {
    auto row = ptt_layout[leader];
    std::sort(row.begin(), row.end());
    std::ostream time_output (std::cout.rdbuf());
    std::ostream scalability_output (std::cout.rdbuf());
    std::ostream width_output (std::cout.rdbuf());
    std::ostream empty_output (std::cout.rdbuf());
    time_output  << std::scientific << std::setprecision(3);
    scalability_output << std::setprecision(3);    
    empty_output << std::left << std::setw(5);
    std::cout << "Thread = " << leader << std::endl;    
    std::cout << "==================================" << std::endl;
    std::cout << " | " << std::setw(5) << "Width" << " | " << std::setw(9) << std::left << "Time" << " | " << "Scalability" << std::endl;
    std::cout << "==================================" << std::endl;
    for (int i = 0; i < row.size(); ++i) {
      int curr_width = row[i];
      if(curr_width <= 0) continue;
      auto comp_perf = table[curr_width - 1][leader];
      std::cout << " | ";
      width_output << std::left << std::setw(5) << curr_width;
      std::cout << " | ";      
      time_output << comp_perf; 
      std::cout << " | ";
      if(i == 0) {        
        empty_output << " - ";
      } else if(comp_perf != 0.0f) {
        auto scaling = table[row[0] - 1][leader]/comp_perf;
        auto efficiency = scaling / curr_width;
        if(efficiency  < 0.6 || efficiency > 1.3) {
          scalability_output << "(" <<table[row[0] - 1][leader]/comp_perf << ")";  
        } else {
          scalability_output << table[row[0] - 1][leader]/comp_perf;
        }
      }
      std::cout << std::endl;  
    }
    std::cout << std::endl;
  }
}  

#ifdef ERASE
int PolyTask::globalsearch_Perf(int nthread, PolyTask * it){
  float shortest_exec = 100000.0f;
  float comp_perf = 0.0f; 
  int new_width = 1; 
  int new_leader = -1;
  for(int leader = 0; leader < ptt_layout.size(); ++leader) {
    for(auto&& width : ptt_layout[leader]) {
       auto&& ptt_val = 0.0f;
#if (defined DVFS) && (defined TX2)
      if(leader < 2){
        ptt_val = it->get_timetable(denver_freq, leader, width - 1);
      }else{
        ptt_val = it->get_timetable(a57_freq, leader, width - 1);
      }
#else
      ptt_val = it->get_timetable(0, leader, width-1);
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout <<"[DEBUG] Priority=1, ptt value("<< leader << "," << width << ") = " << ptt_val << std::endl;
      LOCK_RELEASE(output_lck);
#endif
#endif
      if(Sched == 0){
        if(ptt_val == 0.0f) {
          new_width = width;
          new_leader = leader; 
          leader = ptt_layout.size(); 
          break;
        }
      }
      if(Sched == 1){
// #ifdef DVFS
//         if(ptt_val == 0.0f) {
// #else
        // if(ptt_val == 0.0f|| (PTT_UpdateFlag[leader][width-1] <= EAS_PTT_TRAIN)) {
        if(ptt_val == 0.0f) {
// #endif
          new_width = width;
          new_leader = leader; 
          leader = ptt_layout.size(); 
          break;
        }
      }
      
#ifdef CRI_COST
      //For Cost
      comp_perf = width * ptt_val;
#endif
#ifdef CRI_PERF
      //For Performance
      comp_perf = ptt_val;
#endif

      if (comp_perf < shortest_exec) {
        shortest_exec = comp_perf;
        new_width = width;
        new_leader = leader;      
      }
#ifdef EAS
    }
#endif
    }
  }  
  it->width = new_width; 
  it->leader = new_leader;
  it->updateflag = 1;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout <<"[DEBUG] Priority=1, task "<< it->taskid <<" will run on thread "<< it->leader << ", width become " << it->width << std::endl;
  LOCK_RELEASE(output_lck);
#endif
  return new_leader;
}

int PolyTask::ERASE_Target_Perf(int nthread, PolyTask * it){
  float comp_perf = 0.0f;
  int new_width[NUMSOCKETS] = {1};
  int new_leader[NUMSOCKETS] = {0};
#ifndef MultipleKernels
  if(!ptt_full){
#endif
  for(int leader = 0; leader < ptt_layout.size(); ++leader) {
    for(auto&& width : ptt_layout[leader]) {
      auto&& ptt_val = 0.0f;
      ptt_val = it->get_timetable(0, leader, width - 1);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "ptt_val(leader=" << leader <<",width=" << width << ") = " << it->get_timetable(0, leader, width - 1) << "\n";
        LOCK_RELEASE(output_lck);
#endif
      // if(ptt_val == 0.0f || (PTT_UpdateFlag[leader][width-1] <= EAS_PTT_TRAIN)) {
      if(ptt_val == 0.0f){
        it->width  = width;
        it->leader = leader;
        it->updateflag = 1;
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] PTT_Value (" << leader << ", " << width << ") = 0.0f. Run with this configuration." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
        return it->leader;
      }
      else{
        continue;
      }
    }
  }
#ifndef MultipleKernels   
  }
  ptt_full = true;
// #ifdef DynaDVFS
//   if(freq_dec > 0){
//     env = 3;
//     freq_dec = 0;
//   }else{
//     if(freq_inc > 0){
//       env = 0;
//       freq_inc = 0;
//     }
//   }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] PTT is fully trained. freq_dec / freq_inc are set to zeros. " << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
// #endif
#endif
#if (defined AveCluster) && (defined TX2)
  float average[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0}; 
  float ptt_value[XITAO_MAXTHREADS] = {0.0};
  float fastest[NUMSOCKETS] = {0.0}; 
  for(int a = 0; a < NUMSOCKETS; a++){
    float shortest_exec = 100000.0f;
    int start_idle = (a==0)? 0 : 2;
    int end_idle = (a==0)? 2 : gotao_nthreads;
    for(int testwidth = 1; testwidth <= end_idle; testwidth *= 2){
      int same_type = 0;
      for (int h = start_idle; h < end_idle; h += testwidth){
        ptt_value[h] = it->get_timetable(0, h,testwidth-1); // width = 1
        average[a][testwidth] += ptt_value[h];
        same_type++;
      }
      average[a][testwidth] = average[a][testwidth] / float(same_type);
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "Average[" << a << "][" << testwidth << "] = " << average[a][testwidth] << std::endl;
      LOCK_RELEASE(output_lck);
#endif
      comp_perf = average[a][testwidth];
      if (comp_perf < shortest_exec) {
        shortest_exec = comp_perf;
        new_leader[a] = start_idle + (rand() % ((end_idle - start_idle)/testwidth)) * testwidth;
        new_width[a] = testwidth;
        fastest[a] = comp_perf; 
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    if(a==0){
      std::cout << "[DEBUG] Denver fastest is when width = " << new_width << ". PTT value = " << fastest[a] << std::endl;
    }else{
      std::cout << "[DEBUG] A57 fastest is when width = " << new_width << ". PTT value = " << fastest[a] << std::endl;
    }
    LOCK_RELEASE(output_lck);
#endif
  }
#ifdef ERASE_target_perf 
  // The ratio of Denver - fastest to A57 - fastest
  D_A = float(fastest[1] / fastest[0]);
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] The performance ratio of Denver over A57 is " << D_A << "." << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
  if(fastest[1] > fastest[0]){
    it->width = new_width[0];
    it->leader = new_leader[0];
  }else{
    it->width = new_width[1];
    it->leader = new_leader[1];
  }
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] The fastest configuration is (" << it->leader << ", " << it->width << ")." << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
  it->updateflag = 1;
  return it->leader;
}

int PolyTask::ERASE_Target_EDP(int nthread, PolyTask * it){
  float idleP = 0.0f;
  float total_idleP = 0.0f;
  float dyna_P =  0.0f;
  int sum_cluster_active = 0;
  float comp_perf = 0.0f;
#ifndef MultipleKernels
  if(!ptt_full){
#endif
  for(int leader = 0; leader < ptt_layout.size(); ++leader) {
    for(auto&& width : ptt_layout[leader]) {
      auto&& ptt_val = 0.0f;
      ptt_val = it->get_timetable(0, leader, width - 1);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "ptt_val(leader=" << leader <<",width=" << width << ") = " << it->get_timetable(0, leader, width - 1) << "\n";
        LOCK_RELEASE(output_lck);
#endif
      //if(ptt_val == 0.0f || (PTT_UpdateFlag[leader][width-1] <= EAS_PTT_TRAIN)) {
      if(ptt_val == 0.0f){
        it->width  = width;
        it->leader = leader;
        it->updateflag = 1;
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] PTT_Value (" << leader << ", " << width << ") = 0.0f. Run with this configuration." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
        return it->leader;
      }
      else{
        continue;
      }
    }
  }
#ifndef MultipleKernels   
  }
  ptt_full = true;
// #ifdef DynaDVFS
//   if(freq_dec > 0){
//     env = 3;
//     freq_dec = 0;
//   }else{
//     if(freq_inc > 0){
//       env = 0;
//       freq_inc = 0;
//     }
//   }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] PTT is fully trained. freq_dec / freq_inc are set to zeros." << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
// #endif
#endif
#if (defined AveCluster) && (defined TX2)
  float ptt_value[XITAO_MAXTHREADS] = {0.0};
  float fastest[NUMSOCKETS] = {0.0}; 
  int new_width[NUMSOCKETS] = {0};
  int new_leader[NUMSOCKETS] = {0};
  float power_edp[NUMSOCKETS] = {0.0};
  float perf_edp[NUMSOCKETS] = {0.0};
  for(int a = 0; a < NUMSOCKETS; a++){
    float shortest_exec = 100000.0f;
    int start_idle = (a==0)? 0 : 2;
    int end_idle = (a==0)? 2 : gotao_nthreads;
    if(it->tasktype == 0){
#ifdef DynaDVFS
      idleP = compute_bound_power[env][a][0];
      dyna_P = compute_bound_power[env][a][end_idle-start_idle];
#else
      idleP = compute_bound_power[a][0];
      dyna_P = compute_bound_power[a][end_idle-start_idle];
#endif
    }else{
      if(it->tasktype == 1){
#ifdef DynaDVFS
        idleP = memory_bound_power[env][a][0];
        dyna_P = memory_bound_power[env][a][end_idle-start_idle];
#else
        idleP = memory_bound_power[a][0];
        dyna_P = memory_bound_power[a][end_idle-start_idle];
#endif
      }else{
        if(it->tasktype == 2){
#ifdef DynaDVFS
          idleP = cache_intensive_power[env][a][0];
          dyna_P = cache_intensive_power[env][a][end_idle-start_idle];
#else
          idleP = cache_intensive_power[a][0];
          dyna_P = cache_intensive_power[a][end_idle-start_idle];
#endif
        }
      }
    } 
    // sum_cluster_active = std::accumulate(status+ start_idle, status+end_idle, 0);
    // float idleP2 = 0.0f;
    // if(a == 0){
    //   idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[2] || status[3] || status[4] || status[5]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    // }else{
    //   idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[0] || status[1]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    // }
    for(int testwidth = 1; testwidth <= end_idle; testwidth *= 2){
      // sum_cluster_active = (sum_cluster_active < testwidth)? testwidth : sum_cluster_active;
      // idleP = idleP2 * testwidth / sum_cluster_active;
      // if(it->tasktype == 0){
      //   dyna_P = compute_bound_power[a][testwidth];
      // }else{
      //   if(it->tasktype == 1){
      //     dyna_P = memory_bound_power[a][testwidth];
      //   }else{
      //     if(it->tasktype == 2){
      //       dyna_P = cache_intensive_power[a][testwidth];
      //     }
      //   }
      // }
      int parallelism = (end_idle - start_idle) / testwidth;
      int same_type = 0;
      average[a][testwidth] = 0.0;
      for (int h = start_idle; h < end_idle; h += testwidth){
        ptt_value[h] = it->get_timetable(0, h,testwidth-1); // width = 1
        average[a][testwidth] += ptt_value[h];
        same_type++;
      }
      average[a][testwidth] = average[a][testwidth] / float(same_type);
      // Consider task parallelism: divided by ((end_idle - start_idle)/testwidth)
      comp_perf = std::pow(average[a][testwidth]/parallelism, 2.0) * (idleP + dyna_P);
      // comp_perf = average[a][testwidth] * average[a][testwidth] * (idleP + dyna_P) / ((end_idle - start_idle)/testwidth);
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "Average[" << a << "][" << testwidth << "] = " << average[a][testwidth] << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif
      if (comp_perf < shortest_exec) {
        shortest_exec = comp_perf;
        best_leader_config[a] = start_idle + (rand() % ((end_idle - start_idle)/testwidth)) * testwidth;
        best_width_config[a] = testwidth;
        fastest[a] = comp_perf; 
        best_power_config[a] = (idleP + dyna_P)/parallelism;
        best_perf_config[a] = average[a][testwidth];
      }
    }
// #if (defined ERASE_target_edp_method2) 
//     if(a == 0){
//       best_width_config[0] = new_width[a];
//       Denver_best_edp_leader = new_leader[a];
//       best_power_config[0] = power_edp[a];
//       best_perf_config[0] = perf_edp[a];
//     }else{
//       best_width_config[1] = new_width[a];
//       A57_best_edp_leader = new_leader[a];
//       best_power_config[1] = power_edp[a];
//       best_perf_config[1] = perf_edp[a];
//     }
// #endif
  }
#if (defined ERASE_target_edp_method2)
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] Denver best: (" << best_leader_config[0] << ", " << best_width_config[0] << ". Power: " << best_power_config[0] << ". Perf: " << best_perf_config[0] << ". EDP=" << fastest[0] << std::endl;
  std::cout << "[DEBUG] A57 best: (" << best_leader_config[1] << ", " << best_width_config[1] << ". Power: " << best_power_config[1] << ". Perf: " << best_perf_config[1] << ". EDP=" << fastest[1] << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ERASE_target_edp_method1
  // The ratio of Denver - fastest to A57 - fastest
  D_A = float(fastest[1] / fastest[0]);
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] The EDP ratio of Denver over A57 is " << D_A << "." << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
  if(fastest[1] > fastest[0]){
    it->width = best_width_config[0];
    it->leader = best_leader_config[0];
#if (defined ERASE_target_edp_method2)
    if(it->tasktype == 0){
      // Power of A57 = total - Denver power - A57 idle power
      // best_power_config[1] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[0]*(2/best_width_config[0]) - compute_bound_power[1][0]) / (4/best_width_config[1]);
      best_power_config[1] -= compute_bound_power[1][0];
      best_power_config[0] += compute_bound_power[1][0];
    }
#endif
  }else{
    it->width = best_width_config[1];
    it->leader = best_leader_config[1];
#if (defined ERASE_target_edp_method2)
    if(it->tasktype == 0){
      // best_power_config[0] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[1]*(4/best_width_config[1]) - compute_bound_power[0][0]) / (2/best_width_config[0]);
      best_power_config[0] -= compute_bound_power[0][0];
      best_power_config[1] += compute_bound_power[0][0];
    }
#endif
  }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] The fastest configuration is (" << it->leader << ", " << it->width << ")." << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
#endif
  it->updateflag = 1;
  return it->leader;
}
#endif

void PolyTask::frequency_tuning(int nthread, int best_cluster, int freq_index){
  uint64_t bestfreq = avail_freq[freq_index];
  if(best_cluster == 0){ //Denver
    Denver << std::to_string(bestfreq) << std::endl;
    /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
    int cluster_active = std::accumulate(status_working + start_coreid[1], status_working + end_coreid[1], 0);   
    if(cluster_active == 0 && cur_freq_index[1] > cur_freq_index[0]){ /* No working cores on A57 cluster and the current frequency of A57 is higher than working Denver, then tune the frequency */
      ARM << std::to_string(bestfreq) << std::endl;
      cur_freq[1] = bestfreq; /* Update the current frequency */
      cur_freq_index[1] = freq_index;
    }  
  }else{
    ARM << std::to_string(bestfreq) << std::endl;
    /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
    int cluster_active = std::accumulate(status_working + start_coreid[0], status_working + end_coreid[0], 0);   
    if(cluster_active == 0 && cur_freq_index[0] > cur_freq_index[1]){ /* No working cores on Denver cluster and the current frequency of Denver is higher than working A57, then tune the frequency */
      Denver << std::to_string(bestfreq) << std::endl;
      cur_freq[0] = bestfreq; /* Update the current frequency */
      cur_freq_index[0] = freq_index;
    }  
  }
  cur_freq[best_cluster] = bestfreq; /* Update the current frequency */
  cur_freq_index[best_cluster] = freq_index; /* Update the current frequency index */
}

int PolyTask::find_best_config(int nthread, PolyTask * it){ /* The kernel task hasn't got the best config yet, three loops to search for the best configs. */
  float shortest_exec = 100000.0f;
  float energy_pred = 0.0f;
  float idleP_cluster = 0.0f;
  int sum_cluster_active[NUMSOCKETS] = {0};
  for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ /* Get the number of active cores in each cluster */
    sum_cluster_active[clus_id] = std::accumulate(status+start_coreid[clus_id], status+end_coreid[clus_id], 0); 
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] Number of active cores in cluster " << clus_id << ": " << sum_cluster_active[clus_id] << ". status[0] = " << status[0] \
    << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] << std::endl;
    LOCK_RELEASE(output_lck);
#endif 
  }
  /* Step 1: first standard: check if the execution time of (current frequency of Denver, Denver, 2) < 1 ms? --- define as Fine-grained tasks --- Find out best core type, number of cores, doesn't change frequency */
  /* After finding out the best config, double check if it is fine-grained? TBD */
  if(it->get_timetable(cur_freq_index[0], 0, 1) < FINE_GRAIN_THRESHOLD){
    // it->previous_tasktype = it->tasktype; /* First fine-grained task goes here and set its task type as the previous task type for next incoming task */
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
        idleP_cluster = idle_power[cur_freq_index[clus_id]][clus_id] + idle_power[cur_freq_index[1-clus_id]][1-clus_id]; /* Then equals idle power of whole chip */
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
      }else{
        idleP_cluster = idle_power[cur_freq_index[clus_id]][clus_id]; /* otherwise, equals idle power of the cluster */
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
      }
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
        sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
        float idleP = idleP_cluster * wid / sum_cluster_active[clus_id];
        float runtimeP = it->get_powertable(cur_freq_index[clus_id], clus_id, wid-1);
        float timeP = it->get_timetable(cur_freq_index[clus_id], clus_id, wid-1);
        energy_pred = timeP * (runtimeP + idleP); /* it->get_powertable() is only the prediction for runtime power. */
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Frequency: " << cur_freq[clus_id] << ", cluster " << clus_id << ", width "<< wid << ", sum_cluster_active = " \
          << sum_cluster_active[clus_id] << ", idle power " << idleP << ", runtime power " << runtimeP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
          LOCK_RELEASE(output_lck);
#endif 
          if(energy_pred < shortest_exec){
            shortest_exec = energy_pred;
            it->set_best_cluster(clus_id);
            it->set_best_numcores(wid);
          }
        }
      }
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] For current frequency: " << cur_freq[it->get_best_cluster()] << ", best cluster: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << std::endl;
      LOCK_RELEASE(output_lck);
#endif 
      // if(it->get_timetable(cur_freq_index[best_cluster], best_cluster, best_width) < FINE_GRAIN_THRESHOLD){ /* Double confirm that the task using the best config is still fine-grained */
      it->set_enable_freq_change(false); /* No DFS */
      it->granularity_fine = true; /* Mark it as fine-grained task */
      // }else{ /* It is not */
      //   it->granularity_fine = true;
      // }
    }else{ /* Coarse-grained tasks --- Find out best frequency, core type, number of cores */
      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
        for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
          if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
            idleP_cluster = idle_power[freq_indx][clus_id] + idle_power[freq_indx][1-clus_id]; /* Then equals idle power of whole chip */
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
          }else{
            idleP_cluster = idle_power[freq_indx][clus_id]; /* otherwise, equals idle power of the cluster */
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
          }
          for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
            sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
            float idleP = idleP_cluster * wid / sum_cluster_active[clus_id];
            float runtimeP = it->get_powertable(freq_indx, clus_id, wid-1);
            float timeP = it->get_timetable(freq_indx, clus_id, wid-1);
#ifdef EDP_TEST_
            energy_pred = timeP * timeP * (runtimeP + idleP);
#else
            energy_pred = timeP * (runtimeP + idleP); /* it->get_powertable() is only the prediction for runtime power. */
#endif
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Frequency: " << avail_freq[freq_indx] << ", cluster " << clus_id << ", width "<< wid << ", sum_cluster_active = " \
            << sum_cluster_active[clus_id] << ", idle power " << idleP << ", runtime power " << runtimeP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
            if(energy_pred < shortest_exec){
              shortest_exec = energy_pred;
              it->set_best_freq(freq_indx);
              it->set_best_cluster(clus_id);
              it->set_best_numcores(wid);
            }
          }
        }
      }
      it->set_enable_freq_change(true);
// #ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] " << it->kernel_name << ": the optimal frequency: " << avail_freq[it->get_best_freq()] << ", best cluster: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << std::endl;
      LOCK_RELEASE(output_lck);
// #endif 
    }
    it->set_bestconfig_state(true);
    it->get_bestconfig = true; // the reason of adding this: check if task itself gets the best config or not, since not all tasks are released because of dependencies (e.g., alya, k-means, dot product)
    it->width = it->get_best_numcores();
    int best_cluster = it->get_best_cluster();
    it->leader = start_coreid[best_cluster] + (rand() % ((end_coreid[best_cluster] - start_coreid[best_cluster])/it->width)) * it->width;
    return it->leader;
}

/* After the searching, the kernel task got the best configs: best frequency, best cluster and best width, new incoming tasks directly use the best config. */
int PolyTask::update_best_config(int nthread, PolyTask * it){ 
  it->width = it->get_best_numcores();
  int best_cluster = it->get_best_cluster();
  it->leader = start_coreid[best_cluster] + (rand() % ((end_coreid[best_cluster] - start_coreid[best_cluster])/it->width)) * it->width;
  it->get_bestconfig = true;
  if(it->get_timetable(cur_freq_index[best_cluster], best_cluster, it->width - 1) < FINE_GRAIN_THRESHOLD){ /* Incoming tasks are fine-grained */ 
    it->granularity_fine = true; /* Mark it as fine-grained task */
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    if(it->get_enable_freq_change() == false){
      std::cout << "[DEBUG] For fine-grained task " << it->taskid << ", current frequency: " << cur_freq[it->get_best_cluster()] << ", best cluster: " << it->get_best_cluster() \
      << ", best width: " << it->get_best_numcores() << ", it->get_timetable(cur_freq_index[best_cluster], best_cluster, it->width - 1) = " \
      << it->get_timetable(cur_freq_index[best_cluster], best_cluster, it->width - 1) << std::endl;
    }else{
      std::cout << "[DEBUG] For fine-grained task " << it->taskid << ", BEST frequency: " << avail_freq[it->get_best_freq()] << ", best cluster: " << it->get_best_cluster() \
      << ", best width: " << it->get_best_numcores() << ", it->get_timetable(cur_freq_index[best_cluster], best_cluster, it->width - 1) = " \
      << it->get_timetable(cur_freq_index[best_cluster], best_cluster, it->width - 1) << std::endl;
    }
    LOCK_RELEASE(output_lck);
#endif 
  }
  return it->leader;
}

// Targeting Energy by sharing workload across clusters
int PolyTask::ERASE_Target_Energy_2(int nthread, PolyTask * it){
  float comp_perf = 0.0f;
  if(it->tasktype < num_kernels){
  // if(it->get_timetable_state(2) == false){     /* PTT training is not finished yet */
  if(global_training == false){
    for(int cluster = 0; cluster < NUMSOCKETS; ++cluster) {
      for(auto&& width : ptt_layout[start_coreid[cluster]]) {
        auto&& ptt_val = 0.0f;
        ptt_val = it->get_timetable(ptt_freq_index[cluster], cluster, width - 1);
#ifdef TRAIN_METHOD_1 /* Allow three tasks to train the same config, pros: training is faster, cons: not apply to memory-bound tasks */
        if(it->get_PTT_UpdateFlag(ptt_freq_index[cluster], cluster, width-1) < NUM_TRAIN_TASKS){
          it->width  = width;
          it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
          it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << it->kernel_name <<"->Timetable(" << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = " << ptt_val << ". Run with (" << it->leader << ", " << it->width << ")." << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          return it->leader;
        }else{
          continue;
        }  
#endif
#ifdef TRAIN_METHOD_2 /* Allow DOP tasks to train the same config, pros: also apply to memory-bound tasks, cons: training might be slower */
        if(ptt_val == 0.0f){ 
          it->width  = width;
          it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
          // it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << it->kernel_name <<"->Timetable(" << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = 0. Run with (" << it->leader << ", " << it->width << ")." << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          return it->leader;
        }else{
          continue;
        }  
#endif
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] Not fully trained. ptt_val(" << cluster <<", " << width << ") = " << ptt_val << "\n";
//         LOCK_RELEASE(output_lck);
// #endif
//         if(ptt_val > 0.0f && PTT_UpdateFlag[ptt_freq_index[cluster]][cluster][width-1] < 3){
//           it->width  = width;
//           it->leader = start_coreid + (rand() % ((end_coreid - start_coreid)/width)) * width;
//           PTT_UpdateFlag[ptt_freq_index[cluster]][cluster][width-1]++;
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] PTT(" << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") > 0. PTT_Finsh < 3. Run with (" << it->leader << ", " << it->width << ")."  << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
//           return it->leader;
//         }else{
//           if(ptt_val == 0.0f && PTT_UpdateFlag[ptt_freq_index[cluster]][cluster][width-1] < 3){  /* Allow 3 tasks execute with same resource configuration */
//             it->width  = width;
//             it->leader = start_coreid + (rand() % ((end_coreid - start_coreid)/width)) * width;
//             PTT_UpdateFlag[ptt_freq_index[cluster]][cluster][width-1]++;
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] PTT(" << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = 0. Run with (" << it->leader << ", " << it->width << ")."  << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif
//             return it->leader;
//           }
//           else{
//             continue;
//           }
//         }
      }
    }
    /* There have been enough tasks travesed the PTT and train the PTT, but still training phase hasn't been finished yet, for the incoming tasks, execution config goes to here. 
    Method: schedule incoming tasks to random cores with random width */ 
    if(rand() % gotao_nthreads < START_A){ // Schedule to Denver
      it->width = pow(2, rand() % 2); // Width: 1 2
      it->leader = START_D + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    }else{ // Schedule to A57
      it->width = pow(2, rand() % 3); // Width: 1 2 4
      it->leader = START_A + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Run with (" << it->leader << ", " << it->width << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
    return it->leader;
  }else{ /* PTT is fully trained */
  if(it->get_bestconfig_state() == false){ /* The kernel task hasn't got the best config yet, three loops to search for the best configs. */
    it->find_best_config(nthread, it);
  }else{ /* After the searching, the kernel task got the best configs: best frequency, best cluster and best width, new incoming tasks directly use the best config. */
    it->update_best_config(nthread, it);
//       for(int i = start_coreid[best_cluster]; i < end_coreid[best_cluster]; i++){ 
//       }
//       int tot_fine_grained = consecutive_fine_grained[it->tasktype][best_cluster]++;
//       if(tot_fine_grained * it->get_timetable(cur_freq_index[0], 0, 1) > FINE_GRAIN_THRESHOLD){ /* Enough fine-grained tasks => coarse-grained task */
//         // it->get_bestconfig_state(false); /* Method 1: search for the new best config for the next incoming tasks (not for this one): best freq, cluster, #cores */
//         // it->set_enable_freq_change(true); /* Enable frequency change */
//         /* Below the codes: Find out the best freq, cluster, #cores */
//         float shortest_exec = 100000.0f;
//         float energy_pred = 0.0f;
//         float idleP_cluster = 0.0f;
//         int sum_cluster_active[NUMSOCKETS] = {0};
//         for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ /* Get the number of active cores in each cluster */
//           sum_cluster_active[clus_id] = std::accumulate(status+start_coreid[clus_id], status+end_coreid[clus_id], 0); 
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] Number of active cores in cluster " << clus_id << ": " << sum_cluster_active[clus_id] << ". status[0] = " << status[0] \
//           << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif 
//         }
//         for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
//           for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
//             if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
//               idleP_cluster = idle_power[freq_indx][clus_id] + idle_power[freq_indx][1-clus_id]; /* Then equals idle power of whole chip */
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif 
//             }else{
//               idleP_cluster = idle_power[freq_indx][clus_id]; /* otherwise, equals idle power of the cluster */
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif 
//             }
//             for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
//               sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
//               float idleP = idleP_cluster * wid / sum_cluster_active[clus_id];
//               float runtimeP = it->get_powertable(freq_indx, clus_id, wid-1);
//               float timeP = it->get_timetable(freq_indx, clus_id, wid-1);
//               energy_pred = timeP * (runtimeP + idleP); /* it->get_powertable() is only the prediction for runtime power. */
// // #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Frequency: " << avail_freq[freq_indx] << ", cluster " << clus_id << ", width "<< wid << ", sum_cluster_active = " \
//               << sum_cluster_active[clus_id] << ", idle power " << idleP << ", runtime power " << runtimeP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
//               LOCK_RELEASE(output_lck);
// // #endif 
//               if(energy_pred < shortest_exec){
//                 shortest_exec = energy_pred;
//                 it->set_best_freq(freq_indx);
//                 it->set_best_cluster(clus_id);
//                 it->set_best_numcores(wid);
//               }
//             }
//           }
//         }
//         it->set_bestconfig_state(true);
//         it->width = it->get_best_numcores();
//         int best_cluster = it->get_best_cluster();
//         it->leader = start_coreid[best_cluster] + (rand() % ((end_coreid[best_cluster] - start_coreid[best_cluster])/it->width)) * it->width;
//       }
//     }
  }
#if 0
#if (defined AveCluster) && (defined TX2)
  float ptt_value[XITAO_MAXTHREADS] = {0.0};
  float fastest[NUMSOCKETS] = {0.0}; 
  int new_width[NUMSOCKETS] = {0};
  int new_leader[NUMSOCKETS] = {0};
  float power_edp[NUMSOCKETS] = {0.0};
  float perf_edp[NUMSOCKETS] = {0.0};
  for(int a = 0; a < NUMSOCKETS; a++){
    float shortest_exec = 100000.0f;
    int start_idle = (a==0)? 0 : 2;
    int end_idle = (a==0)? 2 : gotao_nthreads;
    if(it->tasktype == 0){
#ifdef DynaDVFS
      idleP = compute_bound_power[env][a][0];
      dyna_P = compute_bound_power[env][a][end_idle-start_idle];
#else
      idleP = compute_bound_power[a][0];
      dyna_P = compute_bound_power[a][end_idle-start_idle];
#endif
    }else{
      if(it->tasktype == 1){
#ifdef DynaDVFS
        idleP = memory_bound_power[env][a][0];
        dyna_P = memory_bound_power[env][a][end_idle-start_idle];
#else
        idleP = memory_bound_power[a][0];
        dyna_P = memory_bound_power[a][end_idle-start_idle];
#endif
      }else{
        if(it->tasktype == 2){
#ifdef DynaDVFS
          idleP = cache_intensive_power[env][a][0];
          dyna_P = cache_intensive_power[env][a][end_idle-start_idle];
#else
          idleP = cache_intensive_power[a][0];
          dyna_P = cache_intensive_power[a][end_idle-start_idle];
#endif
        }
      }
    } 
    // sum_cluster_active = std::accumulate(status+ start_idle, status+end_idle, 0);
    // float idleP2 = 0.0f;
    // if(a == 0){
    //   idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[2] || status[3] || status[4] || status[5]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    // }else{
    //   idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[0] || status[1]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    // }
    for(int testwidth = 1; testwidth <= end_idle; testwidth *= 2){
      // sum_cluster_active = (sum_cluster_active < testwidth)? testwidth : sum_cluster_active;
      // idleP = idleP2 * testwidth / sum_cluster_active;
      // if(it->tasktype == 0){
      //   dyna_P = compute_bound_power[a][testwidth];
      // }else{
      //   if(it->tasktype == 1){
      //     dyna_P = memory_bound_power[a][testwidth];
      //   }else{
      //     if(it->tasktype == 2){
      //       dyna_P = cache_intensive_power[a][testwidth];
      //     }
      //   }
      // }
      int parallelism = (end_idle - start_idle) / testwidth;
      int same_type = 0;
      average[a][testwidth] = 0.0;
      for (int h = start_idle; h < end_idle; h += testwidth){
        ptt_value[h] = it->get_timetable(0, h,testwidth-1); // width = 1
        average[a][testwidth] += ptt_value[h];
        same_type++;
      }
      average[a][testwidth] = average[a][testwidth] / float(same_type);
      // Consider task parallelism: divided by ((end_idle - start_idle)/testwidth)
      comp_perf = average[a][testwidth]/parallelism * (idleP + dyna_P);
      // comp_perf = average[a][testwidth] * average[a][testwidth] * (idleP + dyna_P) / ((end_idle - start_idle)/testwidth);
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "Average[" << a << "][" << testwidth << "] = " << average[a][testwidth] << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif
      if (comp_perf < shortest_exec) {
        shortest_exec = comp_perf;
        best_leader_config[a] = start_idle + (rand() % ((end_idle - start_idle)/testwidth)) * testwidth;
        best_width_config[a] = testwidth;
        fastest[a] = comp_perf; 
        best_power_config[a] = (idleP + dyna_P)/parallelism;
        best_perf_config[a] = average[a][testwidth];
      }
    }
  }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] For task " << it->taskid << ": Denver best: (" << best_leader_config[0] << ", " << best_width_config[0] << "). Power: " << best_power_config[0] << ". Perf: " << best_perf_config[0] << ". Energy=" << fastest[0] << std::endl;
//   std::cout << "[DEBUG] For task " << it->taskid << ": A57 best: (" << best_leader_config[1] << ", " << best_width_config[1] << "). Power: " << best_power_config[1] << ". Perf: " << best_perf_config[1] << ". Energy=" << fastest[1] << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
  if(fastest[1] > fastest[0]){
    best_cluster_config = 0;
    second_best_cluster_config = 1;
/*    if(it->tasktype == 0){
      // Power of A57 = total - Denver power - A57 idle power
      // best_power_config[1] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[0]*(2/best_width_config[0]) - compute_bound_power[1][0]) / (4/best_width_config[1]);
      best_power_config[1] -= compute_bound_power[1][0];
      best_power_config[0] += compute_bound_power[1][0];
    }
    if(it->tasktype == 1){
      // Power of A57 = total - Denver power - A57 idle power
      // best_power_config[1] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[0]*(2/best_width_config[0]) - compute_bound_power[1][0]) / (4/best_width_config[1]);
      best_power_config[1] -= memory_bound_power[1][0];
      best_power_config[0] += memory_bound_power[1][0];
    }
    if(it->tasktype == 2){
      // Power of A57 = total - Denver power - A57 idle power
      // best_power_config[1] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[0]*(2/best_width_config[0]) - compute_bound_power[1][0]) / (4/best_width_config[1]);
      best_power_config[1] -= cache_intensive_power[1][0];
      best_power_config[0] += cache_intensive_power[1][0];
    }*/
  }else{
    best_cluster_config = 1;
    second_best_cluster_config = 0;
// #if (defined ERASE_target_edp_method2) || (defined ERASE_target_test)
/*    if(it->tasktype == 0){
      // best_power_config[0] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[1]*(4/best_width_config[1]) - compute_bound_power[0][0]) / (2/best_width_config[0]);
      best_power_config[0] -= compute_bound_power[0][0];
      best_power_config[1] += compute_bound_power[0][0];
    }
    if(it->tasktype == 1){
      // Power of A57 = total - Denver power - A57 idle power
      // best_power_config[1] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[0]*(2/best_width_config[0]) - compute_bound_power[1][0]) / (4/best_width_config[1]);
      best_power_config[0] -= memory_bound_power[0][0];
      best_power_config[1] += memory_bound_power[0][0];
    }
    if(it->tasktype == 2){
      // Power of A57 = total - Denver power - A57 idle power
      // best_power_config[1] = (compute_bound_power[0][gotao_nthreads-1] - best_power_config[0]*(2/best_width_config[0]) - compute_bound_power[1][0]) / (4/best_width_config[1]);
      best_power_config[0] -= cache_intensive_power[0][0];
      best_power_config[1] += cache_intensive_power[0][0];
    }*/
// #endif
  }
    it->width = best_width_config[best_cluster_config];
    it->leader = best_leader_config[best_cluster_config];
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] For task " << it->taskid << ": Denver best: (" << best_leader_config[0] << ", " << best_width_config[0] << "). Power: " << best_power_config[0] << ". Perf: " << best_perf_config[0] << ". Energy=" << fastest[0] << std::endl;
    std::cout << "[DEBUG] For task " << it->taskid << ": A57 best: (" << best_leader_config[1] << ", " << best_width_config[1] << "). Power: " << best_power_config[1] << ". Perf: " << best_perf_config[1] << ". Energy=" << fastest[1] << std::endl;
    std::cout << "[DEBUG] Task "<< it->taskid <<" executes with ("<< it->leader << ", " << it->width << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
#endif

#endif
  }
  }else{
    if(rand() % gotao_nthreads < START_A){ // Schedule to Denver
      it->width = pow(2, rand() % 2); // Width: 1 2
      it->leader = START_D + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    }else{ // Schedule to A57
      it->width = pow(2, rand() % 3); // Width: 1 2 4
      it->leader = START_A + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Run with (" << it->leader << ", " << it->width << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
  }
  return it->leader;
}

#ifdef ERASE
int PolyTask::ERASE_Target_Energy(int nthread, PolyTask * it){
#ifdef PTTaccuracy
  float pred_time = 0.0f;
#endif
  float shortest_exec = 100000.0f;
  float comp_perf = 0.0f;
  int new_width = 1;
  int new_leader = 0;
#ifdef second_efficient
  float second_shortest_exec = 100000.0f;
  int previous_new_width = 1; 
  int previous_new_leader = -1;
#endif
#ifdef DVFS
  int new_freq = -1; 
#endif
  auto idleP = 0.0f;
  float dyna_P = 0;
#ifdef ACCURACY_TEST
  auto new_idleP = 0.0f;
  float new_dynaP = 0;
#endif
  int start_idle, end_idle, sum_cluster_active = 0; //real_p = 0;
  auto&& partitions = inclusive_partitions[nthread];

/* Firstly, fill out PTT table */
#ifndef MultipleKernels
  if(!ptt_full){
#endif
#ifdef DVFS
  for(int freq = 0; freq < FREQLEVELS; freq++){
#endif
    for(int leader = 0; leader < ptt_layout.size(); ++leader) {
       for(auto&& width : ptt_layout[leader]) {
        auto&& ptt_val = 0.0f;
#ifdef DVFS
        ptt_val = it->get_timetable(freq, leader, width - 1);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "ptt_val(freq=" << freq << ",leader=" << leader <<",width=" << width << ") = " << it->get_timetable(freq, leader, width - 1) << "\n";
        LOCK_RELEASE(output_lck);
#endif
#else
        ptt_val = it->get_timetable(0, leader, width - 1);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "ptt_val(leader=" << leader <<",width=" << width << ") = " << it->get_timetable(0, leader, width - 1) << "\n";
        LOCK_RELEASE(output_lck);
#endif
#endif
// #ifdef DVFS
//         if((ptt_val == 0.0f)  || (PTT_UpdateFlag[fr][leader][width-1] <= EAS_PTT_TRAIN)) {
// #else
//         if(ptt_val == 0.0f|| (PTT_UpdateFlag[leader][width-1] <= EAS_PTT_TRAIN)) {
// #endif
        if(ptt_val == 0.0f){
          it->width  = width;
          it->leader = leader;
          it->updateflag = 1;
#ifdef DVFS
          new_freq = freq;
          freq = FREQLEVELS;
#endif
#if (defined TX2) && (defined DVFS)
          // Change frequency
          if(it->leader < 2){
            // first check if the frequency is the same? If so, go ahead; if not, need to change frequency!
            if(new_freq != denver_freq){
              std::cout << "[DVFS] Denver frequency changes from = " << denver_freq << " to " << new_freq << ".\n";
              // Write new frequency to Denver cluster
              if(new_freq == 0){
                system("echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
              }
              if(new_freq == 1){
                system("echo 345600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
              }
              denver_freq = new_freq;
            }
          }else{
            // Write new frequency to A57 cluster
            if(new_freq != a57_freq){
              std::cout << "[DVFS] A57 frequency changes from = " << a57_freq << " to " << new_freq << ".\n";
              if(new_freq == 0){
                system("echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
              }
              if(new_freq == 1){
                system("echo 345600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
              }
              a57_freq = new_freq;
            }
          }
#endif
#if (defined Energyaccuracy) && (defined TX2) && (defined MultipleKernels)
          int a = (leader < 2)? 0: 1;
          idleP =(a==0)? compute_bound_power[0][0] : compute_bound_power[1][0];
          start_idle = (a==0)? 0 : 2;
          end_idle = (a==0)? 2 : gotao_nthreads;
          sum_cluster_active = std::accumulate(status+ start_idle, status+end_idle, 0);
          float idleP2 = 0.0f;
          if(a == 0){
            idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[2] || status[3] || status[4] || status[5]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
          }else{
            idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[0] || status[1]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
          }
          sum_cluster_active = (sum_cluster_active < width)? width : sum_cluster_active;
          idleP = idleP2 * width / sum_cluster_active;
          dyna_P = compute_bound_power[a][width];
          if(ptt_val != 0.0f){
            it->finalenergypred = ptt_val *(idleP + dyna_P);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Task " << it->taskid <<" ptt value = " << ptt_val << ", Predicted energy / power =  " << it->finalenergypred << ". \n";
            LOCK_RELEASE(output_lck);
#endif
          }else{
            it->finalenergypred = 0.0f;
            it->finalpowerpred = idleP + dyna_P;
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Task " << it->taskid <<" ptt value = " << ptt_val << ", Predicted  power =  " << it->finalpowerpred << ". \n";
            LOCK_RELEASE(output_lck);
#endif
          }

#endif
          //leader = ptt_layout.size();
          // PTT_Count++;
          // break;
          return it->leader;
        }
        else{
          continue;
        }
        //std::cout << "After break, in poly task " << it->taskid << ", it->width = " << it->width << ", it->leader = " << it->leader << "\n";
      }
    }
#ifndef MultipleKernels
    std::chrono::time_point<std::chrono::system_clock> start_ERASE;
    start_ERASE = std::chrono::system_clock::now();
    auto start_ERASE_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(start_ERASE);
    auto epoch = start_ERASE_ms.time_since_epoch();
    std::cout << "ERASE PTT Full Time (For ERASE Accuracy Purpose):" << epoch.count() << std::endl;
    std::cout << "\n";
#endif
    // std::cout << "After warming up PTT:\n";
    // for(int leader = 0; leader < ptt_layout.size(); ++leader) {
    //   for(auto&& width : ptt_layout[leader]) {
    //     std::cout << "ptt_val(leader=" << leader <<",width=" << width << ") = " << it->get_timetable( leader, width - 1) << "\n";
    //   }
    // }
#ifdef DVFS
  }  
#endif
#ifndef MultipleKernels
  }
  ptt_full = true;
// #ifdef DynaDVFS
//   if(freq_dec > 0){
//     env = 3;
//     freq_dec = 0;
//   }else{
//     if(freq_inc > 0){
//       env = 0;
//       freq_inc = 0;
//     }
//   }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] PTT is fully trained. freq_dec / freq_inc are set to zeros. " << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
// #endif
#endif

/* Secondly, search for energy efficient one */
#if (defined AveCluster) && (defined TX2)
  float average[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0}; 
  float ptt_value[XITAO_MAXTHREADS] = {0.0};
  int count = 0;
  for(int a = 0; a < NUMSOCKETS; a++){
    if(it->tasktype == 0){
#if (defined DVFS)
      idleP =(leader < 2)? compute_bound_power[denver_freq][0][0] : compute_bound_power[a57_freq][1][0];
#elif (defined DynaDVFS)
      idleP =(leader < 2)? compute_bound_power[env][0][0] : compute_bound_power[env][1][0];
#else
      idleP =(leader < 2)? compute_bound_power[0][0] : compute_bound_power[1][0];
#endif
    }else{
      if(it->tasktype == 1){
#ifdef DVFS
        idleP =(leader < 2)? memory_bound_power[denver_freq][0][0] : memory_bound_power[a57_freq][1][0];
#elif (defined DynaDVFS)
        idleP =(leader < 2)? memory_bound_power[env][0][0] : memory_bound_power[env][1][0];
#else
        idleP =(leader < 2)? memory_bound_power[0][0] : memory_bound_power[1][0];
#endif
      }else{
        if(it->tasktype == 2){
#ifdef DVFS
          idleP =(leader < 2)? cache_intensive_power[denver_freq][0][0] : cache_intensive_power[a57_freq][1][0];
#elif (defined DynaDVFS)
          idleP =(leader < 2)? cache_intensive_power[env][0][0] : cache_intensive_power[env][1][0];
#else
          idleP =(leader < 2)? cache_intensive_power[0][0] : cache_intensive_power[1][0];
#endif
        }
      }
    }
    
    start_idle = (a==0)? 0 : 2;
    end_idle = (a==0)? 2 : gotao_nthreads;
    sum_cluster_active = std::accumulate(status+ start_idle, status+end_idle, 0);
    float idleP2 = 0.0f;
    if(a == 0){
      idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[2] || status[3] || status[4] || status[5]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    }else{
      idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[0] || status[1]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    }

    for(int testwidth = 1; testwidth <= end_idle; testwidth*=2){
      count++;
      sum_cluster_active = (sum_cluster_active < testwidth)? testwidth : sum_cluster_active;
      idleP = idleP2 * testwidth / sum_cluster_active;
      if(it->tasktype == 0){
#if (defined DVFS)
        dyna_P = compute_bound_power[a57_freq][a][testwidth];
#elif (defined DynaDVFS)
        dyna_P = compute_bound_power[env][a][testwidth];
#else
        dyna_P = compute_bound_power[a][testwidth];
#endif
      }else{
        if(it->tasktype == 1){
#if (defined DVFS)
          dyna_P = memory_bound_power[a57_freq][a][testwidth];
#elif (defined DynaDVFS)
          dyna_P = memory_bound_power[env][a][testwidth];
#else
          dyna_P = memory_bound_power[a][testwidth];
#endif  
        }else{
          if(it->tasktype == 2){
#if (defined DVFS)
            dyna_P = cache_intensive_power[a57_freq][a][testwidth];
#elif (defined DynaDVFS)
            dyna_P = cache_intensive_power[env][a][testwidth];
#else
            dyna_P = cache_intensive_power[a][testwidth];
#endif
          }
        }
      }
      int same_type = 0;
      for (int h = start_idle; h < end_idle; h+=testwidth){
        ptt_value[h] = it->get_timetable(0, h,testwidth-1); // width = 1
        average[a][testwidth] += ptt_value[h];
        same_type++;
      }
      average[a][testwidth] = average[a][testwidth] / float(same_type);

      comp_perf = average[a][testwidth] *(idleP + dyna_P);
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] a = " << a << ". width = " << testwidth << ". Average PTT = " << average[a][testwidth] << ". idleP = " << idleP << ". Dyna_P = " << dyna_P << ". Energy prediction = " << comp_perf << std::endl;
      LOCK_RELEASE(output_lck);
#endif

      if (comp_perf < shortest_exec) {
#ifdef PTTaccuracy
        pred_time = average[a][testwidth];
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Global1: Task " << it->taskid << " time prediction: " << pred_time << "\n";
        LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef second_efficient
        if(count > 1){
          second_shortest_exec = shortest_exec;
          previous_new_leader = new_leader;
          previous_new_width = new_width;
 #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Cluster " << a << ". Previous best leader is " << previous_new_leader << ", best width is " << previous_new_width << std::endl;
          LOCK_RELEASE(output_lck);
#endif       
        }
#endif
        //new_leader = start_idle + rand() % (end_idle - start_idle);
        shortest_exec = comp_perf;
        new_leader = start_idle + (rand() % ((end_idle - start_idle)/testwidth)) * testwidth;
        new_width = testwidth;
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Cluster " << a << ". Current best leader is " << new_leader << ", best width is " << new_width << std::endl;
        LOCK_RELEASE(output_lck);
#endif
      }
#ifdef second_efficient
      else{
        if(comp_perf < second_shortest_exec){
          second_shortest_exec = comp_perf;
          previous_new_leader = start_idle + (rand() % ((end_idle - start_idle)/testwidth)) * testwidth;
          previous_new_width = testwidth;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Cluster " << a << ". Previous best leader is " << previous_new_leader << ", best width is " << previous_new_width << std::endl;
          LOCK_RELEASE(output_lck);
#endif
        }
      }
#endif
    }
  }
#else

#ifdef DVFS
  for(int freq = 0; freq < FREQLEVELS; freq++){
#endif
  for(int leader = 0; leader < ptt_layout.size(); ++leader) {
#if defined(TX2)
      //idleP =(leader < 2)? 76 : 152; //mWatt
      if(it->tasktype == 0){
#ifdef DVFS
        idleP =(leader < 2)? compute_bound_power[denver_freq][0][0] : compute_bound_power[a57_freq][1][0];
#else
        idleP =(leader < 2)? compute_bound_power[0][0] : compute_bound_power[1][0];
#endif
      }
      if(it->tasktype == 1){
#ifdef DVFS
        idleP =(leader < 2)? memory_bound_power[denver_freq][0][0] : memory_bound_power[a57_freq][1][0];
#else
        idleP =(leader < 2)? memory_bound_power[0][0] : memory_bound_power[1][0];
#endif
      }
      start_idle = (leader < 2)? 0 : 2;
      end_idle = (leader < 2)? 2 : gotao_nthreads;
#endif

#if defined(Haswell)
      start_idle = (leader < gotao_nthreads/num_sockets)? 0 : gotao_nthreads/num_sockets;
      end_idle = (leader < gotao_nthreads/num_sockets)? gotao_nthreads/num_sockets : gotao_nthreads;
#endif
      sum_cluster_active = std::accumulate(status+ start_idle, status+end_idle, 0);

      // Real Parallelism (Dynamic Power sharing, for COPY case)
      //real_p = (Parallelism < sum_cluster_active) ? Parallelism : sum_cluster_active;

#if defined(TX2)
      float idleP2 = 0;
      if(leader < 2){
        //idleP2 = 228 - (status[2] || status[3] || status[4] || status[5]) * (228-idleP);
        idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[2] || status[3] || status[4] || status[5]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
      }else{
        //idleP2 = 228 - (status[0] || status[1]) * (228-idleP);
        idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[0] || status[1]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
      }
#endif
      for(auto&& width : ptt_layout[leader]) {
        auto&& ptt_val = 0.0f;
#ifdef DVFS
        ptt_val = it->get_timetable(freq, leader, width - 1);
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "ptt_val(freq=" << freq << ",leader=" << leader <<",width=" << width << ") = " << it->get_timetable(freq, leader, width - 1) << "\n";
//         LOCK_RELEASE(output_lck);
// #endif
        //if((ptt_val == 0.0f) || (PTT_UpdateFlag[freq][leader][width-1] <= EAS_PTT_TRAIN)){   
#else
        ptt_val = it->get_timetable(0, leader, width - 1);
        //if((ptt_val == 0.0f) || (PTT_UpdateFlag[leader][width-1] <= EAS_PTT_TRAIN)){
#endif
//         if(ptt_val == 0.0f) {
//           new_width = width;
//           new_leader = leader;
// #ifdef DVFS
//           new_freq = freq;
//           freq = FREQLEVELS;
// #endif
//           leader = ptt_layout.size();
//           break;
//         }
        /********* ********* Idle Power ********* *********/
        int num_active = 0;
        for(int i = leader; i < leader+width; i++){
          if(status[i] == 0){
            num_active++;
          }
        }
#ifdef TX2
        idleP = idleP2 * width / (sum_cluster_active + num_active);
#endif

#ifdef Haswell
        //idleP = idleP * width / (sum_cluster_active + num_active);
        idleP = power[leader/COREPERSOCKET][0] * width / (sum_cluster_active + num_active);
#endif 

        /********* ********* Dynamic Power ********* *********/
#ifdef TX2
        // int real_core_use = (((real_p + num_active) * width) < (end_idle - start_idle)) ? ((real_p + num_active) * width) : (end_idle - start_idle);
        // int real_use_bywidth = real_core_use / width;
        // dyna_P = it->get_power(leader, real_core_use, real_use_bywidth); 
#ifdef DVFS
        // MM tasks
        if(it->tasktype == 0){
          if(leader < 2){
            dyna_P = compute_bound_power[denver_freq][0][width];
          }
          else{
            dyna_P = compute_bound_power[a57_freq][1][width];
          }
        }else{
          if(it->tasktype == 1){
            if(leader < 2){
              dyna_P = memory_bound_power[denver_freq][0][width];
            }
            else{
              dyna_P = memory_bound_power[a57_freq][1][width];
            }
          }
        }         
        
#else
        if(it->tasktype == 0){
          if(leader < 2){
            dyna_P = compute_bound_power[0][width];
          }
          else{
            dyna_P = compute_bound_power[1][width];
          }
        }
        else{
          if(it->tasktype == 1){
            if(leader < 2){
              dyna_P = memory_bound_power[0][width];
            }
            else{
              dyna_P = memory_bound_power[1][width];
            }
          }
        }
#endif
#endif

#if defined(Haswell)
        //dyna_P = it->Haswell_Dyna_power(leader, width);
        dyna_P = power[leader/COREPERSOCKET][width];
#endif
        /********* ********* Energy Prediction ********* *********/
        // Prediction is based on idle and dynamic power
        comp_perf = ptt_val * (idleP + dyna_P);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << " Prediction time = " << ptt_val << ", idle Power =  " << idleP << ", dynamic Power =  " << dyna_P << ". Energy = " << comp_perf << std::endl;
        LOCK_RELEASE(output_lck);
#endif       
        // Prediction is based on only dynamic power
        //comp_perf = ptt_val * dyna_P;
				if (comp_perf < shortest_exec) {
#ifdef DVFS
          new_freq = freq;
#endif
#ifdef second_efficient
          previous_new_width = new_width;
          previous_new_leader = new_leader;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "Previous best leader is " << previous_new_leader << ", best width is " << previous_new_width << std::endl;
          LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef PTTaccuracy
          pred_time = float(ptt_val);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Global1: Task " << it->taskid << " time prediction: " << pred_time << "\n";
          LOCK_RELEASE(output_lck);
#endif
#endif
          shortest_exec = comp_perf;
          new_width = width;
          new_leader = leader;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "Current best leader is " << new_leader << ", best width is " << new_width << std::endl;
          LOCK_RELEASE(output_lck);
#endif
#ifdef ACCURACY_TEST
          new_dynaP = dyna_P;
          new_idleP = idleP;
#endif
        }
		  }
	  }
#ifdef DVFS
  }
#endif
// #ifdef TX2
// }
#endif
#ifdef second_efficient
    it->width = previous_new_width;
    it->leader = previous_new_leader;
    it->updateflag = 1;
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] For task " << it->taskid << ", SECOND energy efficient choce: it->width = " << previous_new_width << ", it->leader = " << previous_new_leader << ". Most energy efficient choce: it->width = " << new_width << ", it->leader = " << new_leader <<"\n";
    LOCK_RELEASE(output_lck);
#endif
#else
  it->width = new_width;
  it->leader = new_leader;
  it->updateflag = 1;
#ifdef PTTaccuracy
  it->finalpredtime = pred_time;
#endif
#ifdef Energyaccuracy
  it->finalenergypred = shortest_exec;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "Task " << it->taskid <<" predicted energy =  " << it->finalenergypred << ". \n";
  LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] For task " << it->taskid << ", most energy efficient choce: it->width = " << it->width << ", it->leader = " << it->leader << "\n";
  LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ACCURACY_TEST
  LOCK_ACQUIRE(output_lck);
  std::chrono::time_point<std::chrono::system_clock> start;
  start = std::chrono::system_clock::now();
  auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(start);
  auto epoch1 = start1_ms.time_since_epoch();
  std::cout << "Task " << it->taskid << ", " << epoch1.count() << ", idle power: " << new_idleP << ", dynamic power: " << new_dynaP << ", total: " << new_idleP + new_dynaP << std::endl;
  LOCK_RELEASE(output_lck);
#endif
  //std::cout << "After break, in poly task " << it->taskid << ", it->width = " << it->width << ", it->leader = " << it->leader << "\n";
#if (defined TX2) && (defined DVFS)
  // Change frequency
  if(it->leader < 2){
    // first check if the frequency is the same? If so, go ahead; if not, need to change frequency!
    if(new_freq != denver_freq){
      std::cout << "[DVFS] Denver frequency changes from = " << denver_freq << " to " << new_freq << ".\n";
      // Write new frequency to Denver cluster
      
      if(new_freq == 0){
        int status = system("echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
        if (status < 0){
          std::cout << "Error: " << strerror(errno) << '\n';
        }
      }
      if(new_freq == 1){
        int status = system("echo 345600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
        if (status < 0){
          std::cout << "Error: " << strerror(errno) << '\n';
        }
      }
      denver_freq = new_freq;
    }
  }else{
    // Write new frequency to A57 cluster
    if(new_freq != a57_freq){
      std::cout << "[DVFS] A57 frequency changes from = " << a57_freq << " to " << new_freq << ".\n";
      if(new_freq == 0){
        system("echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
      }
      if(new_freq == 1){
        system("echo 345600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
      }
      a57_freq = new_freq;
    }
  }
#endif
// #ifdef second_efficient
//   return previous_new_leader;
// #else
//   return new_leader;
// #endif

  return it->leader;
}
//#endif

int PolyTask::eas_width_mold(int _nthread, PolyTask *it){
#ifdef PTTaccuracy
  float pred_time = 0.0f;
#endif
#ifdef second_efficient
  int previous_new_width = 1; 
  int previous_new_leader = -1;
#endif
  int new_width = 1; 
  int new_leader = -1;
#ifdef DVFS
  int new_freq = -1; 
#endif
  float shortest_exec = 1000.0f;
  float comp_perf = 0.0f; 
  auto&& partitions = inclusive_partitions[_nthread];
  auto idleP = 0.0f;
  float dyna_P = 0;
  int start_idle, end_idle, sum_cluster_active = 0; //real_p = 0;

  if(!ptt_full){
#ifdef DVFS
  for(int freq = 0; freq < FREQLEVELS; freq++){
#endif
  for(auto&& elem : partitions) {
    int leader = elem.first;
    int width  = elem.second;
    auto&& ptt_val = 0.0f;
#ifdef DVFS
    ptt_val = it->get_timetable(freq, leader, width - 1);
    // if((ptt_val == 0.0f) || (PTT_UpdateFlag[freq][leader][width-1] <= EAS_PTT_TRAIN)) 
    if(ptt_val == 0.0f)
#else
    ptt_val = it->get_timetable(0, leader, width - 1);
    if(ptt_val == 0.0f)
    // if((ptt_val == 0.0f) || (PTT_UpdateFlag[leader][width-1] <= EAS_PTT_TRAIN)) 
#endif
    {
      it->width = width;
      it->leader = leader;    
      it->updateflag = 1;
#ifdef DVFS  
      new_freq = freq; 
      freq = FREQLEVELS;
#endif
#ifdef DVFS
      // Change frequency
      if(it->leader < 2){
        // first check if the frequency is the same? If so, go ahead; if not, need to change frequency!
        if(new_freq != denver_freq){
          std::cout << "[DVFS] Denver frequency changes from = " << denver_freq << " to " << new_freq << ".\n";
          // Write new frequency to Denver cluster
          if(new_freq == 0){
            system("echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
          }else{
            system("echo 345600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
          }
          denver_freq = new_freq;
        }
      }else{
        // Write new frequency to A57 cluster
        if(new_freq != a57_freq){
          std::cout << "[DVFS] A57 frequency changes from = " << a57_freq << " to " << new_freq << ".\n";
          if(new_freq == 0){
            system("echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
          }else{
            system("echo 345600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
          }
          a57_freq = new_freq;
        }
      }
#endif
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] After Stealing, ptt(" << leader << "," << width << ") = 0, go with this one. \n";
      LOCK_RELEASE(output_lck);
#endif
      return leader;
    }
    else{
      continue;
    }
  }
#ifdef DVFS
  }
#endif
  }
  ptt_full = true;
// #ifdef DynaDVFS
//   if(freq_dec > 0){
//     env = 3;
//     freq_dec = 0;
//   }else{
//     if(freq_inc > 0){
//       env = 0;
//       freq_inc = 0;
//     }
//   }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] PTT is fully trained. freq_dec / freq_inc are set to zeros. " << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
// #endif
#ifdef DVFS
  for(int freq = 0; freq < FREQLEVELS; freq++){
#endif
  for(auto&& elem : partitions) {
    int leader = elem.first;
    int width  = elem.second;
    auto&& ptt_val = 0.0f;
#ifdef DVFS
    ptt_val = it->get_timetable(freq, leader, width - 1);
#else
    ptt_val = it->get_timetable(0, leader, width - 1);
#endif
#if defined(TX2)
    //idleP =(leader < 2)? 76 : 152; //mWatt
    if(it->tasktype == 0){
#if (defined DVFS)
      idleP =(leader < 2)? compute_bound_power[denver_freq][0][0] : compute_bound_power[a57_freq][1][0];
#elif (defined DynaDVFS)
      idleP =(leader < 2)? compute_bound_power[env][0][0] : compute_bound_power[env][1][0];
#else
      idleP =(leader < 2)? compute_bound_power[0][0] : compute_bound_power[1][0];
#endif
    }else{
      if(it->tasktype == 1){
#ifdef DVFS
        idleP =(leader < 2)? memory_bound_power[denver_freq][0][0] : memory_bound_power[a57_freq][1][0];
#elif (defined DynaDVFS)
        idleP =(leader < 2)? memory_bound_power[env][0][0] : memory_bound_power[env][1][0];
#else
        idleP =(leader < 2)? memory_bound_power[0][0] : memory_bound_power[1][0];
#endif
      }else{
        if(it->tasktype == 2){
#ifdef DVFS
          idleP =(leader < 2)? cache_intensive_power[denver_freq][0][0] : cache_intensive_power[a57_freq][1][0];
#elif (defined DynaDVFS)
          idleP =(leader < 2)? cache_intensive_power[env][0][0] : cache_intensive_power[env][1][0];
#else
          idleP =(leader < 2)? cache_intensive_power[0][0] : cache_intensive_power[1][0];
#endif
        }
      }
    }
    
    start_idle = (leader < 2)? 0 : 2;
    end_idle = (leader < 2)? 2 : gotao_nthreads;
#endif

#if defined(Haswell)
    start_idle = (leader < gotao_nthreads/num_sockets)? 0 : gotao_nthreads/num_sockets;
    end_idle = (leader < gotao_nthreads/num_sockets)? gotao_nthreads/num_sockets : gotao_nthreads;
#endif
    sum_cluster_active = std::accumulate(status+ start_idle, status+end_idle, 0);

    //real_p = (Parallelism < sum_cluster_active) ? Parallelism : sum_cluster_active;

#if defined(TX2)
    float idleP2 = 0;
    if(leader < 2){
      //idleP2 = 228 - (status[2] || status[3] || status[4] || status[5]) * (228-idleP);
      idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[2] || status[3] || status[4] || status[5]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    }else{
      //idleP2 = 228 - (status[0] || status[1]) * (228-idleP);
      idleP2 = (76 + 152 + 76 * ~(a57_freq)) - (status[0] || status[1]) * ((76 + 152 + 76 * ~(a57_freq))-idleP);
    }
#endif

    /********* ********* Idle Power ********* *********/
    int num_active = 0;
    for(int i = leader; i < leader+width; i++){
      if(status[i] == 0){
        num_active++;
      }
    }
#if defined(TX2)
    idleP = idleP2 * width / (sum_cluster_active + num_active);
#endif

#if defined(Haswell)
    //idleP = idleP * width / (sum_cluster_active + num_active);
    idleP = power[leader/COREPERSOCKET][0] * width / (sum_cluster_active + num_active);
#endif 

    /********* ********* Dynamic Power ********* *********/
#if defined(TX2)
    // int real_core_use = (((real_p + num_active) * width) < (end_idle - start_idle)) ? ((real_p + num_active) * width) : (end_idle - start_idle);
    // int real_use_bywidth = real_core_use / width;
    // dyna_P = it->get_power(leader, real_core_use, real_use_bywidth); 
#ifdef DVFS
        // MM tasks
        if(it->tasktype == 0){
          if(leader < 2){
            dyna_P = compute_bound_power[denver_freq][0][width];
          }
          else{
            dyna_P = compute_bound_power[a57_freq][1][width];
          }
        }else{
          if(it->tasktype == 1){
            if(leader < 2){
              dyna_P = memory_bound_power[denver_freq][0][width];
            }
            else{
              dyna_P = memory_bound_power[a57_freq][1][width];
            }
          }
        }         
        
#else
        if(it->tasktype == 0){
#if !defined DVFS  && !defined DynaDVFS
          if(leader < 2){
            dyna_P = compute_bound_power[0][width];
          }
          else{
            dyna_P = compute_bound_power[1][width];
          }
#endif
        }
        else{
          if(it->tasktype == 1){
#if !defined DVFS  && !defined DynaDVFS
            if(leader < 2){
              dyna_P = memory_bound_power[0][width];
            }
            else{
              dyna_P = memory_bound_power[1][width];
            }
#endif
          }
        }
#endif
#endif

#if defined(Haswell)
    //dyna_P = it->Haswell_Dyna_power(leader, width);
    dyna_P = power[leader/COREPERSOCKET][width];
#endif
    /********* ********* Energy Prediction ********* *********/
    // Prediction is based on idle and dynamic power
    comp_perf = ptt_val * (idleP + dyna_P);
    if (comp_perf < shortest_exec) {
#ifdef DVFS
      new_freq = freq;
#endif
#ifdef second_efficient
      previous_new_width = new_width;
      previous_new_leader = new_leader;
#endif
      shortest_exec = comp_perf;
      new_width = width;
      new_leader = leader;      
#ifdef PTTaccuracy
      pred_time = ptt_val;
#endif
    }
  }
#ifdef DVFS
  }
#endif
  if(new_leader != -1) {
#ifdef second_efficient
    it->leader = previous_new_leader;
    it->width  = previous_new_width;
    it->updateflag = 1;
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] After stealing: SECOND energy efficient config: leader = " << it->leader << ", width = " << it->width << ". Most energy efficient leader = " << new_leader << ", width = " << new_width << std::endl;
    LOCK_RELEASE(output_lck);
#endif
#else
    it->leader = new_leader;
    it->width  = new_width;
    it->updateflag = 1;
#ifdef PTTaccuracy
    it->finalpredtime = pred_time;
#endif
#endif
#if (defined DEBUG) && (defined PTTaccuracy)
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] After stealing: Task " << it->taskid << " time prediction: " << it->finalpredtime << "\n";
  LOCK_RELEASE(output_lck);
#endif
  }
#ifdef DVFS
  // Change frequency
  if(it->leader < 2){
    // first check if the frequency is the same? If so, go ahead; if not, need to change frequency!
    if(new_freq != denver_freq){
      std::cout << "[DVFS] Denver frequency changes from = " << denver_freq << " to " << new_freq << ".\n";
      // Write new frequency to Denver cluster
      if(new_freq == 0){
        system("echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
      }else{
        system("echo 345600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
      }
      denver_freq = new_freq;
    }
  }else{
    // Write new frequency to A57 cluster
    if(new_freq != a57_freq){
      std::cout << "[DVFS] A57 frequency changes from = " << a57_freq << " to " << new_freq << ".\n";
      if(new_freq == 0){
        system("echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
      }else{
        system("echo 345600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
      }
      a57_freq = new_freq;
    }
  }
#endif
// #ifdef second_efficient
//   return previous_new_leader;
// #else
//   return new_leader;
// #endif
  return it->leader;
} 
#endif

#ifdef OVERHEAD_PTT
// std::tuple <int, double> 
PolyTask * PolyTask::commit_and_wakeup(int _nthread, std::chrono::duration<double> elapsed_ptt){
#else
PolyTask * PolyTask::commit_and_wakeup(int _nthread){
#endif 
#ifdef OVERHEAD_PTT
  std::chrono::time_point<std::chrono::system_clock> start_ptt, end_ptt;
  start_ptt = std::chrono::system_clock::now();
#endif
  PolyTask *ret = nullptr;
//   if((Sched == 1) || (Sched == 2)){
//     int new_layer_leader = -1;
//     int new_layer_width = -1;
//     std::list<PolyTask *>::iterator it = out.begin();
//     ERASE_Target_Energy(_nthread, (*it));
//     LOCK_ACQUIRE(worker_lock[(*it)->leader]);
//     worker_ready_q[(*it)->leader].push_back(*it);
//     LOCK_RELEASE(worker_lock[(*it)->leader]);
//     new_layer_leader = (*it)->leader;
//     new_layer_width = (*it)->width;
//     ++it;
//     for(it; it != out.end(); ++it){
//       (*it)->width = new_layer_width;
//       (*it)->leader = new_layer_leader;
//       LOCK_ACQUIRE(worker_lock[new_layer_leader]);
//       worker_ready_q[new_layer_leader].push_back(*it);
//       LOCK_RELEASE(worker_lock[new_layer_leader]);
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout <<"[DEBUG] Task "<< (*it)->taskid <<" will run on thread "<< (*it)->leader << ", width become " << (*it)->width << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif
//     }
// 	}
//   else{
    // std::cout << "Thread " << _nthread << " out.size = " << out.size() << std::endl;
#ifdef ERASE_target_edp_method1
    D_give_A = (out.size() > 0) && (ptt_full)? ceil(out.size() * 1 / (D_A+1))+1 : 0; 
#endif

#if (defined ERASE_target_edp_method2)
// || (defined ERASE_target_energy_method2)
    int n = (out.size() > 0) && (ptt_full)? out.size() : 0; 
    D_give_A = 0;
    float standard = 1.0;
    float edp_test = 0.0;
    int temp_width = 0;
    if(n > 0){
#ifdef ERASE_target_energy_method2
      float upper = best_power_config[best_cluster_config] * best_perf_config[best_cluster_config] * n;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[1] The best energy when only running on cluster " << best_cluster_config << ": " << upper << std::endl;
      LOCK_RELEASE(output_lck); 
#endif
#endif
      // int start_idle[2] = {0,2};
      // int end_idle[2] = {2, gotao_nthreads};
      int num_cores[2] = {2,4};
      for(int allow_steal = 1; allow_steal <= n/NUMSOCKETS; allow_steal++){
        // float energy_increase = float(allow_steal)/float(n) * float(((best_power_config[1]*best_perf_config[1])/(best_power_config[0]*best_perf_config[0]))-1) + 1;
        // float power_increase = 1 + float(best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1])) / float(best_power_config[0] * (end_idle[0]-start_idle[0])/best_width_config[0]);
#ifdef ERASE_target_energy_method2
        float Total_energy = 0.0;
        // ERASE - Target Energy
        Total_energy = (n-allow_steal) * best_perf_config[best_cluster_config] * best_power_config[best_cluster_config] + allow_steal * best_perf_config[second_best_cluster_config] * best_power_config[second_best_cluster_config];
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[2] Allow_steal = " << allow_steal << ". Total Energy = " << Total_energy <<  ".\n";
        LOCK_RELEASE(output_lck);          
#endif
        if(Total_energy < upper){
          D_give_A = allow_steal;
          upper = Total_energy;
          // temp_width = new_best_width_config;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[3] Current lowest energy = " << upper << ", D_give_A = " << D_give_A << ".\n";
          LOCK_RELEASE(output_lck);          
#endif
        }
#endif
#ifdef ERASE_target_edp_method2
        // Can use max function
        // if((n-allow_steal)*best_perf_config[0] > allow_steal*best_perf_config[1]){
        //   performance_decline = float(n-allow_steal)/float(n);
        //   // performance_decline = float(n - int(allow_steal * float(best_width_config[0]/(end_idle-start_idle))))/float(n);
        // }else{
        //   performance_decline = (allow_steal*best_perf_config[1])/(n*best_perf_config[0]);
        // }
        float performance_decline = 1.0;
        // Problem Solved: new width can not pass to assembly task (Line 1725-1736)
        if(best_width_config[1] * allow_steal < num_cores[1]){
          int ori_best_width = best_width_config[1];
          int new_best_width_config = ceil(num_cores[1]/allow_steal);
          float new_best_power_config = best_power_config[1] * (new_best_width_config/ori_best_width);
          // int same_type = 0;
          // for (int h = 2; h < gotao_nthreads; h += best_width_config[1]){
          //   average[a][best_width_config[1]] += it->get_timetable(h,testwidth-1); // width = 1
          //    += ptt_value[h];
          //   same_type++;
          // }
          // average[a][testwidth] = average[a][testwidth] / float(same_type);
          float new_best_perf_config = average[1][new_best_width_config];
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[0] new_best_width_config= " << new_best_width_config << ". new_best_power_config =  " << new_best_power_config << ". new_best_perf_config = " << new_best_perf_config << std::endl;
          LOCK_RELEASE(output_lck); 
#endif
          int current_perf = ceil(float((n-allow_steal)*best_width_config[0])/float(num_cores[0]));
          int previous_perf = ceil(float(n*best_width_config[0])/float(num_cores[0]));
          int allow_perf = ceil(float(allow_steal*new_best_width_config)/float(num_cores[1]));
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[1] current_perf= " << current_perf << ". previous_perf =  " << previous_perf << ". allow_perf = " << allow_perf << std::endl;
          LOCK_RELEASE(output_lck); 
  #endif
          if(current_perf < previous_perf){
  #ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[2] Denver execution time = " << current_perf*best_perf_config[0] << ". A57 execution time = " << new_best_perf_config*allow_perf << std::endl;
            LOCK_RELEASE(output_lck); 
  #endif     
            if(current_perf*best_perf_config[0] > new_best_perf_config*allow_perf){
              performance_decline = float(current_perf) / float(previous_perf);
            }else{
              performance_decline = (new_best_perf_config* float(allow_perf))/ (best_perf_config[0] * float(previous_perf));
            }
          }
          // float energy_increase = float(current_perf)/float(previous_perf) + float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) / float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          // A57 wrong Energy consumption: float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf))
          // Denver's original energy consumption (maybe wrong): float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf))
          // float energy_increase = 1 + ( (best_power_config[1] * allow_steal * best_perf_config[1]) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          float energy_increase = 1 + ( (new_best_power_config * allow_steal * new_best_perf_config) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ (best_power_config[0] * best_perf_config[0] * n);
          // edp_test = power_increase * pow(performance_decline, 2.0);
          edp_test = energy_increase *  performance_decline;
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          // std::cout << "[2] Denver's Original Energy consumption: " << float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf)) <<std::endl;
          std::cout << "[3] Denver's Original Energy consumption: " << best_power_config[0] * best_perf_config[0] * n << std::endl;
          std::cout << "[4] Denver reduce energy by " << best_power_config[0] * allow_steal * best_perf_config[0] << std::endl;
          //std::cout << "[4] Additional A57 power consumption: " << best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) << ". A57 consumes energy: " << float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) << std::endl; 
          std::cout << "[5] Additional A57 energy consumption: " << (new_best_power_config * allow_steal * new_best_perf_config) << std::endl;
          LOCK_RELEASE(output_lck); 
  #endif        
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << _nthread << " release " << n << " tasks. Allow_steal =  " << allow_steal << ", energy_increase = " << energy_increase << ", perf_decline = " << performance_decline << ". edp_test = " << edp_test << ".\n";
          LOCK_RELEASE(output_lck);          
  #endif
          if(edp_test < standard){
            D_give_A = allow_steal;
            standard = edp_test;
            temp_width = new_best_width_config;
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Current lowest EDP = " << standard << ", D_give_A = " << D_give_A << ".\n";
          LOCK_RELEASE(output_lck);          
  #endif
          }
        }else{
          int current_perf = ceil(float((n-allow_steal)*best_width_config[0])/float(num_cores[0]));
          int previous_perf = ceil(float(n*best_width_config[0])/float(num_cores[0]));
          int allow_perf = ceil(float(allow_steal*best_width_config[1])/float(num_cores[1]));
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[1] current_perf= " << current_perf << ". previous_perf =  " << previous_perf << ". allow_perf = " << allow_perf << std::endl;
          LOCK_RELEASE(output_lck); 
#endif
          if(current_perf < previous_perf){
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[2] Denver execution time = " << current_perf*best_perf_config[0] << ". A57 execution time = " << best_perf_config[1]*allow_perf << std::endl;
            LOCK_RELEASE(output_lck); 
#endif           

            if(current_perf*best_perf_config[0] > best_perf_config[1]*allow_perf){
              performance_decline = float(current_perf) / float(previous_perf);
            }else{
              performance_decline = (best_perf_config[1]* float(allow_perf))/ (best_perf_config[0] * float(previous_perf));
            }

          }
          // float energy_increase = float(current_perf)/float(previous_perf) + float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) / float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          // A57 wrong Energy consumption: float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf))
          // Denver's original energy consumption (maybe wrong): float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf))
          // float energy_increase = 1 + ( (best_power_config[1] * allow_steal * best_perf_config[1]) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          float energy_increase = 1 + ( (best_power_config[1] * allow_steal * best_perf_config[1]) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ (best_power_config[0] * best_perf_config[0] * n);
          // edp_test = power_increase * pow(performance_decline, 2.0);
          edp_test = energy_increase *  performance_decline;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          // std::cout << "[2] Denver's Original Energy consumption: " << float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf)) <<std::endl;
          std::cout << "[3] Denver's Original Energy consumption: " << best_power_config[0] * best_perf_config[0] * n << std::endl;
          std::cout << "[4] Denver reduce energy by " << best_power_config[0] * allow_steal * best_perf_config[0] << std::endl;
          //std::cout << "[4] Additional A57 power consumption: " << best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) << ". A57 consumes energy: " << float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) << std::endl; 
          std::cout << "[5] Additional A57 energy consumption: " << (best_power_config[1] * allow_steal * best_perf_config[1]) << std::endl;
          LOCK_RELEASE(output_lck); 
#endif        
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << _nthread << " release " << n << " tasks. Allow_steal =  " << allow_steal << ", energy_increase = " << energy_increase << ", perf_decline = " << performance_decline << ". edp_test = " << edp_test << ".\n";
          LOCK_RELEASE(output_lck);          
#endif
          if(edp_test < standard){
            D_give_A = allow_steal;
            standard = edp_test;
            temp_width = best_width_config[1];
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Current lowest EDP = " << standard << ", D_give_A = " << D_give_A << ".\n";
          LOCK_RELEASE(output_lck);          
#endif
          }
        }
        best_width_config[1] = temp_width;
#endif
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] D_give_A = " << D_give_A << ". \n";
        LOCK_RELEASE(output_lck);          
#endif
      }
    }
#endif

    for(std::list<PolyTask *>::iterator it = out.begin(); it != out.end(); ++it){
      int refs = (*it)->refcount.fetch_sub(1);
      if(refs == 1){
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] " << (*it)->kernel_name << " task " << (*it)->taskid << " became ready" << std::endl;
        LOCK_RELEASE(output_lck);
#endif  
        (*it)->updateflag = 0;
#ifdef ERASE
        if(Sched == 0){
          int pr = if_prio(_nthread, (*it));
          if (pr == 1){
#ifdef CATS
            int i = rand() % 2;
            LOCK_ACQUIRE(worker_lock[i]);
            worker_ready_q[i].push_back(*it);
            LOCK_RELEASE(worker_lock[i]);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout <<"[CATS] Priority=1, task "<< (*it)->taskid <<" is pushed to WSQ of thread "<< i << std::endl;
            LOCK_RELEASE(output_lck);
#endif
#else
// Parallelism Sensitivity test for critical tasks according to real time online active cores
#if (defined PARA_TEST) & (defined Haswell)
            if(rand()%100 == 0){
              std::vector<int> list{1, 2, 5, 10};
              int index = rand() % list.size();
              (*it)->width = list[index];
            }else{
              int sum_active = std::accumulate(status_working + 0, status_working + gotao_nthreads, 0);
              if(sum_active < gotao_nthreads - 10){
                (*it)->width = 10;
              }else{
                if((gotao_nthreads - 10) <= sum_active < gotao_nthreads - 5){
                  (*it)->width = 5;
                }else{
                  if((gotao_nthreads - 5) <= sum_active < gotao_nthreads - 2){
                    (*it)->width = 2;
                  }else{
                    (*it)->width = 1;
                  }
                }
              }
            }
            (*it)->leader = _nthread / (*it)->width * (*it)->width;
            LOCK_ACQUIRE(worker_lock[(*it)->leader]);
            worker_ready_q[(*it)->leader].push_back(*it);
            LOCK_RELEASE(worker_lock[(*it)->leader]);
#else
// Else choose the fastest core (Perf) or minimum cost core (Cost)
            globalsearch_Perf(_nthread, (*it));
            for(int i = (*it)->leader; i < (*it)->leader + (*it)->width; i++){
              LOCK_ACQUIRE(worker_assembly_lock[i]);
              worker_assembly_q[i].push_back((*it));
#ifdef NUMTASKS_MIX
              num_task[(*it)->tasktype][ (*it)->width * gotao_nthreads + i]++;
#endif
            }
            for(int i = (*it)->leader; i < (*it)->leader + (*it)->width; i++){
              LOCK_RELEASE(worker_assembly_lock[i]);
            }
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Distributing assembly task " << (*it)->taskid << " with width " << (*it)->width << " to workers [" << (*it)->leader << "," << (*it)->leader + (*it)->width << ")" << std::endl;
            LOCK_RELEASE(output_lck);
#endif  
#endif 
#endif
          }
          else{
#ifdef CATS
            int i = 2 + rand() % 4;
            LOCK_ACQUIRE(worker_lock[i]);
            worker_ready_q[i].push_back(*it);
            LOCK_RELEASE(worker_lock[i]);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout <<"[CATS] Priority=0, task "<< (*it)->taskid <<" is pushed to WSQ of thread "<< i << std::endl;
            LOCK_RELEASE(output_lck);
#endif
#else
            history_mold(_nthread,(*it));
            LOCK_ACQUIRE(worker_lock[_nthread]);
            worker_ready_q[_nthread].push_back(*it);
            LOCK_RELEASE(worker_lock[_nthread]);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Distributing assembly task " << (*it)->taskid << " with width " << (*it)->width << " to workers [" << (*it)->leader << "," << (*it)->leader + (*it)->width << ")" << std::endl;
            LOCK_RELEASE(output_lck);
#endif  
#endif
          }
        }
#endif
        /* Scheduler: ERASE */
        if((Sched == 1) || (Sched == 2)){
#ifdef ERASE_target_energy_method1
          /* Method 1: Fix best config to certain cluster */ 
          // ERASE_Target_Energy(_nthread, (*it));
#endif
#ifdef ERASE_target_energy_method2
          /* Method 2: Allow work stealing across clusters */ 
          // if(D_give_A > 0 && std::distance(out.begin(), it) >= out.size() - D_give_A){
          //   (*it)->width = best_width_config[second_best_cluster_config];
          //   // (*it)->leader = A57_best_edp_leader;
          //   if((*it)->width == 4){
          //     (*it)->leader = 2;
          //   }
          //   if((*it)->width <= 2){
          //     (*it)->leader = 2 + 2 * rand()%2;
          //   }
          //   LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          //   worker_ready_q[(*it)->leader].push_back(*it);
          //   LOCK_RELEASE(worker_lock[(*it)->leader]);
          // }else{
          ERASE_Target_Energy_2(_nthread, (*it));
          // LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          // worker_ready_q[(*it)->leader].push_back(*it);
          // LOCK_RELEASE(worker_lock[(*it)->leader]);
          // } 

          // D_give_A = 2;
          // if((*it)->get_bestconfig_state()==true && std::distance(out.begin(), it) >= out.size() - D_give_A){
          //   (*it)->leader = 2;   
          //   (*it)->width =  4;
          // }
          // else{
          //   (*it)->leader = 0;
          //   (*it)->width = 2;
          // }
          LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          worker_ready_q[(*it)->leader].push_back(*it);
          LOCK_RELEASE(worker_lock[(*it)->leader]);

#endif

#ifdef ERASE_target_perf
          ERASE_Target_Perf(_nthread, (*it));
          LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          worker_ready_q[(*it)->leader].push_back(*it);
          LOCK_RELEASE(worker_lock[(*it)->leader]);
#endif

#if (defined ERASE_target_edp_method1) || (defined ERASE_target_edp_method2)
          if(D_give_A > 0 && std::distance(out.begin(), it) >= out.size() - D_give_A){
            (*it)->width = best_width_config[1];
            // (*it)->leader = A57_best_edp_leader;
            if((*it)->width == 4){
              (*it)->leader = 2;
            }
            if((*it)->width <= 2){
              (*it)->leader = 2 + 2 * rand()%2;
            }
            LOCK_ACQUIRE(worker_lock[(*it)->leader]);
            worker_ready_q[(*it)->leader].push_back(*it);
            LOCK_RELEASE(worker_lock[(*it)->leader]);
          }else{
            ERASE_Target_EDP(_nthread, (*it));
            LOCK_ACQUIRE(worker_lock[(*it)->leader]);
            worker_ready_q[(*it)->leader].push_back(*it);
            LOCK_RELEASE(worker_lock[(*it)->leader]);
          }     
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout <<"[DEBUG] EDP: task "<< (*it)->taskid <<"'s leader id = "<< (*it)->leader << ", width = " << (*it)->width << ", type is " << (*it)->tasktype << std::endl;
          LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ACCURACY_TEST
          for(int i = (*it)->leader; i < (*it)->leader + (*it)->width; i++){
            LOCK_ACQUIRE(worker_assembly_lock[i]);
            worker_assembly_q[i].push_back((*it));
          }
          for(int i = (*it)->leader; i < (*it)->leader + (*it)->width; i++){
            LOCK_RELEASE(worker_assembly_lock[i]);
          }  
#endif

// #ifdef ERASE_target_edp_method2

// #endif          
        }

      /* Scheduler: Random Work Stealing */
			if(Sched == 3){
//          if(!ret && (((*it)->affinity_queue == -1) || (((*it)->affinity_queue/(*it)->width) == (_nthread/(*it)->width)))){
//            ret = *it; // forward locally only if affinity matches
//          }
//          else{
//            int ndx = (*it)->affinity_queue;
//            if((ndx == -1) || (((*it)->affinity_queue/(*it)->width) == (_nthread/(*it)->width)))
//              ndx = _nthread;
//              LOCK_ACQUIRE(worker_lock[ndx]);
//              worker_ready_q[ndx].push_back(*it);
//              LOCK_RELEASE(worker_lock[ndx]);
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout <<"[RWS] Task "<< (*it)->taskid <<" is pushed to WSQ of thread "<< ndx << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif
//          } 

        int ndx = 0;
        // Case 1: EDP Test for A57 borrow n tasks
/*        if(std::distance(out.begin(), it) >= out.size() - 1){
          ndx = 2;
          (*it)->leader = 2;
          (*it)->width = 4;
        }else{
          ndx = std::distance(out.begin(), it) % 2;
          (*it)->leader = ndx;
          (*it)->width = 2;
        } 
        // Case 2: EDP Test for not borrowing task to A57 
        // ndx = std::distance(out.begin(), it) % 2;
        // (*it)->leader = ndx;
        // (*it)->width = 1;
*/
        LOCK_ACQUIRE(worker_lock[ndx]);
        worker_ready_q[ndx].push_back(*it);
        LOCK_RELEASE(worker_lock[ndx]);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout <<"[RWS] Task "<< (*it)->taskid <<" is pushed to WSQ of thread "<< ndx << std::endl;
        LOCK_RELEASE(output_lck);
#endif
			}
    }
  }
#ifdef OVERHEAD_PTT
  end_ptt = std::chrono::system_clock::now();
  elapsed_ptt += end_ptt - start_ptt;
  return elapsed_ptt.count();
#endif
}       
