// tao.h - Task Assembly Operator

#ifndef _TAO_H
#define _TAO_H

#include <sched.h>
#include <unistd.h>
#include <thread>
#include <iostream>
#include "lfq-fifo.h"
#include "config.h"
// #include "xitao_api.h"
#include "poly_task.h"
#include "barriers.h"
#include "xitao_workspace.h"
#include <atomic>
#include <chrono>
#include <functional>
using namespace xitao;

#define GET_TOPO_WIDTH_FROM_LEVEL(x) gotao_sys_topo[x]
typedef void (*task)(void *, int);
#define TASK_SIMPLE   0x0
#define TASK_ASSEMBLY 0x1

// the base class for assemblies is very simple. It just provides base functionality for derived
// classes. The sleeping barrier is used by TAO to synchronize the start of assemblies
class AssemblyTask: public PolyTask{
public:
  AssemblyTask(int w, int nthread=0) : PolyTask(TASK_ASSEMBLY, nthread) {
    leader = nthread-nthread % w;
    width = w;
#ifdef NEED_BARRIER
    barrier = new BARRIER(w);
#endif 
  }
#ifdef NEED_BARRIER
  BARRIER *barrier;
#endif  
  virtual void execute(int thread) = 0;
  ~AssemblyTask(){
#ifdef NEED_BARRIER
    delete barrier;
#endif
  }  
};

class SimpleTask: public PolyTask{
public:
  SimpleTask(task fn, void *a, int nthread=0) : PolyTask(TASK_SIMPLE, nthread), args(a), f(fn){ 
    width = 1; 
  }
  void *args;
  task f;
};

template<class F>
struct xitao_looper {
  F fn;
  void operator()(int start, int end, int thread) const {
    for (int i = start; i < end; ++i)
     fn(i, thread);    
  }
};

template<class F>
xitao_looper<F> looper(F f) {
  return {std::move(f)};
}


// a "ParForTask" that executes a partition on an SPMD region with either dynamic or static scheduling
template <typename FuncType, typename IterType>
class ParForTask: public AssemblyTask {  
private:
  int const _sched_type;
  IterType _start;
  IterType _end;
  FuncType  _spmd_region;  
  IterType _block_size; 
  IterType _block_iter;
  IterType _blocks; 
  IterType _size; 
  std::atomic<int> next_block; /*!< TAO implementation specific atomic variable to provide thread safe tracker of the number of processed blocks */
  const size_t slackness = 8;
public: 
#ifdef DVFS
  static float time_table[][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
#else
  static float time_table[][XITAO_MAXTHREADS];
#endif
  ParForTask(int sched, IterType start, IterType end, FuncType spmd, int width): 
                AssemblyTask(width), _sched_type(sched), _start(start), _end(end), _spmd_region(spmd) { 
    _size = _end - _start; 
    if(_size <= 0) {
      //std::cout << "Error: no work to be done" << std::endl;
      //exit(0);
     _block_iter = 0;
      next_block = 0;
      _blocks = 0; 
      return;
    }
    _block_iter = 0;
    next_block = 0;
    switch(_sched_type) {
      case xitao_vec_static: break;
      case xitao_vec_dynamic: 
        _block_size = _size / (width * slackness);
        if(_block_size <= 0) _block_size = 1;
        _blocks = (_size + _block_size -1) / _block_size;
      break;
      default:
        std::cout << "Error: undefined sched type in XiTAO vector code" << std::endl;
        exit(0);
    }
  }

  void execute(int thread) {   
    if(_sched_type == xitao_vec_dynamic) {
      int block_id = next_block++;
      while(block_id < _blocks) {
        int local_block_start = _start + block_id * _block_size;
        int local_block_end   = (block_id >= _blocks - 1)? _end : local_block_start + _block_size;
        _spmd_region(local_block_start, local_block_end, thread);
        block_id = next_block++;
      }
    } else { // xitao_vec_static*/      
      _block_size = _size / width;
      if(_block_size < 1) {
//        std::cout << "Error: not enough work to do" << std::endl;
  //      exit(0);
          return;
      }      
      int thread_id = thread - leader;
      int local_block_start = _start + thread_id * _block_size;
      int local_block_end   = (thread_id >= width - 1)? _end : local_block_start + _block_size;
      _spmd_region(local_block_start, local_block_end, thread);
    }
  } 

  // int set_timetable(int threadid, float ticks, int index) {
  //   time_table[index][threadid] = ticks;
  // }

  // float get_timetable(int threadid, int index) { 
  //   float time=0;
  //   time = time_table[index][threadid];
  //   return time;
  // }
  
  // Compute-bound
  float get_power(int thread, int real_core_use, int real_use_bywidth) {
    float compute_denver[2][2] = {2046, 2046, 228, 152};
    float compute_a57[2][2] = {988, 809, 76, 76};
    float total_power = 0;
    if(thread < 2){
      total_power = (compute_denver[denver_freq][0] + compute_denver[denver_freq][1] * (real_core_use - 1)) / real_use_bywidth;
    }else{
      total_power = (compute_a57[a57_freq][0] + compute_a57[a57_freq][1] * (real_core_use - 1)) / real_use_bywidth;
    }
    return total_power;
  }
  
  // Cache sensitive
  /*
  float get_power(int thread, int real_core_use, int real_use_bywidth) {
    //float denver[2][2] = {2430, 2195, 228, 228};
    //float a57[2][2] = {1141, 758, 76, 76};
    float denver[2][2] = {2125, 1962, 190, 190};
    float a57[2][2] = {1139, 756, 76, 76};
    float total_power = 0;
    if(thread < 2){
      total_power = (denver[denver_freq][0] + denver[denver_freq][1] *  (real_core_use - 1)) / real_use_bywidth;
    }else{
      total_power = (a57[a57_freq][0] + a57[a57_freq][1] * (real_core_use - 1)) / real_use_bywidth;
    }
    return total_power;
  }

  float get_power(int thread, int real_core_use, int real_use_bywidth) {
    float denver[2][2] = {1590, 1356, 304, 304};
    float a57[2][2] = {985, 327, 76, 57};
    float total_dyna_power = 0;
    if(thread < 2){ 
      total_dyna_power = (denver[denver_freq][0] + denver[denver_freq][1] * (real_core_use - 1)) / real_use_bywidth;
    }else{
      total_dyna_power = (a57[a57_freq][0] + a57[a57_freq][1] * (real_core_use - 1)) / real_use_bywidth;
    }
    return total_dyna_power;
  }
*/
  int cleanup() { }
};

#endif // _TAO_H
