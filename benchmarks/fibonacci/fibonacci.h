#include <iostream>
#include <chrono>
#include <omp.h>
#include <math.h>
#include "xitao.h"
using namespace std;

// enable a known trick to avoid redundant recursion for evaluated cases
//#define MEMOIZE
// the maximum number of Fibonacci terms that can fit in unsigned 64 bit
const uint32_t MAX_FIB = 92;

// a global variable to manage the granularity of TAO creation (coarsening level)
uint32_t grain_size;

// declare the class
class FibTAO;
// init the memoization array of TAOs 
FibTAO* fib_taos[MAX_FIB + 1];

// basic Fibonacci implementation
size_t fib(uint32_t num) {
	// return 0 for 0 and negative terms (undefined)
	if(num <= 0) return 0; 
	// return 1 for the term 1
	else if(num == 1) return 1;
	// recursively find the result
	return fib(num - 1) + fib(num - 2);
}

// basic Fibonacci implementation
size_t fib_omp(uint32_t num) {
	// return 0 for 0 and negative terms (undefined)
	if(num <= 0) return 0; 
	// return 1 for the term 1
	else if(num == 1) return 1;
	// recursively find the result
#pragma omp task if (num > grain_size)
	auto num_1 = fib_omp(num - 1);
#pragma omp task if (num > grain_size)
	auto num_2 = fib_omp(num - 2);
#pragma omp taskwait 
	return num_1 + num_2;
}

// the Fibonacci TAO (Every TAO class must inherit from AssemblyTask)
class FibTAO : public AssemblyTask {
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
	// the n - 1 tao
	FibTAO* prev1;		
	// the n - 2 tao																	
	FibTAO* prev2;
	// the term number																				
	uint32_t term;		
	// the Fib value for the TAO																					
 	size_t val;							
 	// the tao construction. resource hint 1															
 	FibTAO(int _term): term(_term), AssemblyTask(1) { }		
 	// the work function
 	void execute(int nthread) {	
 		// calculate locally if at required granularity
 		if(term <= grain_size) val = fib(term);
 		// if this is not a terminal term													
 		else if(term > 1) 										
 			// calculate the value  												
 			val = prev1->val + prev2->val;
 	}
 	void cleanup(){  }
	
	// Add by Jing
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
};

// build the DAG by reversing the recursion tree 
FibTAO* buildDAG(uint32_t term) {
	// gaurd against negative terms
	if(term <  0) term = 0;
	// if this is terminal term
	if(term <= 1) { 
		// create the terminal tao
		fib_taos[term] = new FibTAO(term);
		fib_taos[term]->tasktype = 0;
		fib_taos[term]->kernel_name = "FibTAO";
		// push the tao
		gotao_push(fib_taos[term]);
		// return the tao
		return fib_taos[term];
	} 
#ifdef MEMOIZE
	// if this TAO has already been created (avoid redundant calculation)
	if(fib_taos[term]) return fib_taos[term];
#endif	
	// construct the tao			
	fib_taos[term] = new FibTAO(term);
	// create TAOs as long as you are above the grain size
	if(term > grain_size) { 
		// build DAG of n - 1 term
		fib_taos[term]->prev1 = buildDAG(term - 1);
		fib_taos[term]->prev1->tasktype = 0;
		fib_taos[term]->prev1->kernel_name = "FibTAO";
		// make edge to current
		fib_taos[term]->prev1->make_edge(fib_taos[term]);
		// build DAG of n - 1 term
		fib_taos[term]->prev2 = buildDAG(term - 2);
		fib_taos[term]->prev2->tasktype = 0;
		fib_taos[term]->prev2->kernel_name = "FibTAO";
		// make edge to current
		fib_taos[term]->prev2->make_edge(fib_taos[term]);
	} else { // you have reached a terminal TAO 
		// push the TAO to fire the DAG execution
		// gotao_push(fib_taos[term]);

		int queue = rand() % gotao_nthreads; //LU0 tasks are randomly executed on Denver
		if(queue < START_A){ // Schedule to Denver
			fib_taos[term]->width = pow(2, rand() % 2); // Width: 1 2
			fib_taos[term]->leader = START_D + (rand() % (2/fib_taos[term]->width)) * fib_taos[term]->width;
		}else{ // Schedule to A57
			fib_taos[term]->width = pow(2, rand() % 3); // Width: 1 2 4
			fib_taos[term]->leader = START_A + (rand() % (4/fib_taos[term]->width)) * fib_taos[term]->width;
		}
		LOCK_ACQUIRE(worker_lock[fib_taos[term]->leader]);
		worker_ready_q[fib_taos[term]->leader].push_back(fib_taos[term]);
		LOCK_RELEASE(worker_lock[fib_taos[term]->leader]);
	}
	// return the current tao (the head of the DAG)
	return fib_taos[term];
}
