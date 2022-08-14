/*! \file 
@brief Defines the basic PolyTask type
*/

#ifndef _POLY_TASK_H
#define _POLY_TASK_H
#include <list>
#include <atomic>
#include <string>
#include "config.h"
#include "lfq-fifo.h"
#include "xitao_workspace.h"
#include "xitao_ptt.h"
extern int a57_freq;
extern int denver_freq;

/*! the basic PolyTask type */
class PolyTask {
public:
  // PolyTasks can have affinity. Currently these are specified on a unidimensional vector
  // space [0,1) of type float. [0,1) are valid affinities, >=1.0 means no affinity
  float affinity_relative_index; 
  // this is the particular queue. When cloning an affinity, we just copy this value
  int   affinity_queue;          
//#if defined(CRIT_PERF_SCHED)
  //Static atomic of current most critical task for criticality-based scheduling
  static std::atomic<int> prev_top_task;     
  //int criticality;
  int marker;
//#endif

  // New code because of running heat
  // A pointer to the corresponding ptt table
  xitao::ptt_shared_type _ptt;  
  // An integer descriptor to distinguish the workload of several TAOs of the same type
  // it is mainly used by the scheduler when picking up the correct PTT
  size_t workload_hint;

  // An string type for indentification of different kernel tasks
  std::string kernel_name;

  int type;
  int tasktype;
  // The leader task id in the resource partition
  int leader;
  long bestfreq;
  long start_running_freq;
  bool granularity_fine;
  int criticality;
  int taskid;
  int updateflag;
  bool get_bestconfig; // check the task gets best config or not
  float temp_ticks[XITAO_MAXTHREADS];

#ifdef PTTaccuracy
  float finalpredtime;
#endif
#ifdef Energyaccuracy
  float finalenergypred;
  float finalpowerpred;
#endif
#if defined(DEBUG)
  static std::atomic<int> created_tasks;
#endif
  static std::atomic<int> pending_tasks;
  
  std::atomic<int> refcount;
  std::list <PolyTask *> out;
  std::atomic<int> threads_out_tao;
  int width; /*!< number of resources that this assembly uses */  

  virtual float get_timetable(int freq_index, int thread, int index) = 0;
  virtual void set_timetable(int freq_index, int thread, float t, int index) = 0;
  virtual float get_powertable(int freq_index, int thread, int index) = 0;
  virtual void set_powertable(int freq_index, int thread, float t, int index) = 0;
  virtual void set_cycletable(int freq_index, int threadid, uint64_t cycles, int index) = 0;
  virtual uint64_t get_cycletable(int freq_index, int threadid, int index) = 0;
  virtual void set_mbtable(int threadid, float mem_b, int index) = 0;
  virtual float get_mbtable(int threadid, int index) = 0;
  virtual bool get_timetable_state(int cluster_index) = 0;
  virtual void set_timetable_state(int cluster_index, bool new_state) = 0;
  virtual bool get_bestconfig_state() = 0;
  virtual void set_bestconfig_state(bool new_state) = 0;
  virtual bool get_enable_freq_change() = 0;
  virtual void set_enable_freq_change(bool new_state) = 0;
  virtual void set_best_freq(int freq_index) = 0;
  virtual void set_best_cluster(int clusterid) = 0;
  virtual void set_best_numcores(int width) = 0;
  virtual int get_best_freq() = 0;
  virtual int get_best_cluster() = 0;
  virtual int get_best_numcores() = 0;
  virtual void increment_PTT_UpdateFinish(int freq_index, int clusterid, int index) = 0;
  virtual float get_PTT_UpdateFinish(int freq_index, int clusterid,int index) = 0;
  virtual void increment_PTT_UpdateFlag(int freq_index, int clusterid, int index) = 0;
  virtual float get_PTT_UpdateFlag(int freq_index, int clusterid,int index) = 0;

  //History-based molding
  virtual int history_mold(int _nthread, PolyTask *it);
  //Recursive function assigning criticality
  int set_criticality();
  int set_marker(int i);
  //Determine if task is critical task
  int if_prio(int _nthread, PolyTask * it);

#ifdef ERASE
  virtual int globalsearch_Perf(int nthread, PolyTask * it);
  virtual int ERASE_Target_Perf(int nthread, PolyTask * it);
  virtual int ERASE_Target_EDP(int nthread, PolyTask * it);
  virtual int ERASE_Target_Energy(int nthread, PolyTask * it);
  virtual int eas_width_mold(int nthread, PolyTask * it);
#endif
  virtual int ERASE_Target_Energy_2(int nthread, PolyTask * it);
  virtual void frequency_tuning(int nthread, int best_cluster, int freq_index);
  virtual int find_best_config(int nthread, PolyTask * it);  // find out the best config for the task type by searching all possible spaces 
  virtual int update_best_config(int nthread, PolyTask * it); // update the best config for upcoming tasks

// #ifdef DVFS
//   static void print_ptt(float table[][XITAO_MAXTHREADS][XITAO_MAXTHREADS], const char* table_name);
// #else
  static void print_ptt(float table[][XITAO_MAXTHREADS], const char* table_name);
// #endif
  //Find suitable thread for prio task
  PolyTask(int t, int _nthread);
  
  //! Convert from an STA to an actual queue number
  /*!
    \param x a floating point value between [0, 1) that indicates the topology address in one dimension
  */
  int sta_to_queue(float x);
  //! give a TAO an STA address
  /*!
    \param x a floating point value between [0, 1) that indicates the topology address in one dimension
  */
  int set_sta(float x);
  //! get the current STA address of a TAO
  float get_sta();
  //! copy the STA of a TAO to the current TAO
  int clone_sta(PolyTask *pt);
  //! create a dependency to another TAO
  /*!
    \param t a TAO with which a happens-before order needs to be ensured (TAO t should execute after *this) 
  */
  void make_edge(PolyTask *t);
  
  //! complete the current TAO and wake up all dependent TAOs
  /*!
    \param _nthread id of the current thread
  */
#ifdef OVERHEAD_PTT
  PolyTask * commit_and_wakeup(int _nthread,  std::chrono::duration<double> elapsed_ptt);
#else
  PolyTask * commit_and_wakeup(int _nthread);
#endif
  
  //! cleanup any dynamic memory that the TAO may have allocated
  virtual void cleanup() = 0;
};
#endif
