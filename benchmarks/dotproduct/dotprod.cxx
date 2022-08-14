/*! @example 
 @brief A program that calculates dotproduct of random two vectors in parallel\n
 we run the example as ./dotprod.out <len> <width> <block> \n
 where
 \param len  := length of vector\n
 \param width := width of TAOs\n
 \param block := how many elements to process per TAO
*/
#include <fstream> 
#include <math.h>
#include "taos_dotproduct.h"
#include "xitao.h"
using namespace xitao;

float VecAdd::time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float VecMulDyn::time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];

float VecAdd::power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float VecMulDyn::power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];

uint64_t VecAdd::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t VecMulDyn::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];

float VecAdd::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
float VecMulDyn::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];

bool VecAdd::time_table_state[NUMSOCKETS+1];
bool VecMulDyn::time_table_state[NUMSOCKETS+1];

bool VecAdd::best_config_state;
bool VecMulDyn::best_config_state;

bool VecAdd::enable_freq_change;
bool VecMulDyn::enable_freq_change;

int VecAdd::best_freqindex;
int VecMulDyn::best_freqindex;

int VecAdd::best_cluster;
int VecMulDyn::best_cluster;

int VecAdd::best_width;
int VecMulDyn::best_width;

std::atomic<int> VecAdd::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> VecMulDyn::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];

std::atomic<int> VecAdd::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> VecMulDyn::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];

const char *scheduler[] = { "PerformanceBased", "EnergyAware", "EDPAware", "RWSS"/* etc */ };

int main(int argc, char *argv[]){
  double *A, *B, *C, D; 
  if(argc != 6) {
    std::cout << "./a.out <schedulerid> <task_iteration> <veclength> <TAOwidth> <blocklength>" << std::endl; 
    return 0;
  }

  int schedulerid = atoi(argv[1]);
  int task_iteration = atoi(argv[2]);
  int len = atoi(argv[3]);
  int width = atoi(argv[4]);
  int block = atoi(argv[5]);

  // For simplicity, only support only perfect partitions
  if(len % block){  
    std::cout << "len is not a multiple of block!" << std::endl;
    return 0;
  }
  std::cout << "---------------------- Test Application - Dot Product ---------------------\n";
  std::cout << "--------- You choose " << scheduler[schedulerid] << " scheduler ---------\n";
  
  //cpu_set_t cpu_;
  //CPU_ZERO(&cpu_);
  //for(int i = 0; i < 8; i+=1) {
  //  CPU_SET(i, &cpu_);
  //} 
  //set_xitao_mask(cpu_);
  
  // no topologies in this version
  A = new double[len];
  B = new double[len];
  C = new double[len];

  // initialize the vectors with some numbers
  srand (time(NULL));
  for(int i = 0; i < len; i++){
    // A[i] = (double) (i+1);
    // B[i] = (double) (i+1);
    A[i] = ((double)rand()) / ((double)RAND_MAX) * 9.9 + 0.1;
    B[i] = ((double)rand()) / ((double)RAND_MAX) * 9.9 + 0.1;
  }

  // init XiTAO runtime 
  // gotao_init();
  gotao_init_hw(-1,-1,-1);
  gotao_init(schedulerid, 1, 0, 0); // 1 Kernel: VecMulDyn
  
  // create numvm TAOs 
  int numvm = len / block;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] Total length = " << len << ", block length = " << block << ", creating " << numvm << " tasks! " << std::endl;
  LOCK_RELEASE(output_lck);
#endif

  // static or dynamic internal TAO scheduler
#ifdef STATIC
  VecMulSta *vm[numvm];  
#else
  VecMulDyn *vm[numvm];
#endif  

  for(int iter = 0; iter < task_iteration; iter++){ // Make more tasks
    VecAdd *start = new VecAdd(C, &D, 0, width);
    start->tasktype = 1;
    start->kernel_name = "VecAdd";
    start->criticality = 1;
    gotao_push(start, 0);

    VecAdd *va = new VecAdd(C, &D, len, width);
    va->tasktype = 1;
    va->kernel_name = "VecAdd";
  // std::cout << "Creating task VecAdd " << va->taskid << std::endl;
  
  // Create the TAODAG
  for(int j = 0; j < numvm; j++){
#ifdef STATIC
    vm[j] = new VecMulSta(A+j*block, B+j*block, C+j*block, block, width);
#else
    vm[j] = new VecMulDyn(A+j*block, B+j*block, C+j*block, block, width);
    vm[j]->tasktype = 0;
    vm[j]->kernel_name = "VecMul";
#endif
    //Create an edge
    vm[j]->make_edge(va);
    start->make_edge(vm[j]);
    //Push current root to assigned queue
    // gotao_push(vm[j], j % gotao_nthreads);

    // int queue = rand() % gotao_nthreads; //LU0 tasks are randomly executed on Denver
    // if(queue < START_A){ // Schedule to Denver
    //   vm[j]->width = pow(2, rand() % 2); // Width: 1 2
    //   vm[j]->leader = START_D + (rand() % (2/vm[j]->width)) * vm[j]->width;
    // }else{ // Schedule to A57
    //   vm[j]->width = pow(2, rand() % 3); // Width: 1 2 4
    //   vm[j]->leader = START_A + (rand() % (4/vm[j]->width)) * vm[j]->width;
    // }

//     for(int cluster = 0; cluster < NUMSOCKETS; ++cluster) {
//       for(auto&& width : ptt_layout[start_coreid[cluster]]) {
//         auto&& ptt_val = 0.0f;
//         ptt_val = it->get_timetable(ptt_freq_index[cluster], cluster, width - 1);
// #ifdef TRAIN_METHOD_1 /* Allow three tasks to train the same config, pros: training is faster, cons: not apply to memory-bound tasks */
//         if(it->get_PTT_UpdateFlag(ptt_freq_index[cluster], cluster, width-1) < NUM_TRAIN_TASKS){
//           it->width  = width;
//           it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
//           it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] " << it->kernel_name <<"->Timetable(" << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = " << ptt_val << ". Run with (" << it->leader << ", " << it->width << ")." << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
//           return it->leader;
//         }else{
//           continue;
//         }  
// #endif
//       }
//     }
    // std::cout << "Creating task VecMul " << vm[j]->taskid << std::endl;
    // LOCK_ACQUIRE(worker_lock[vm[j]->leader]);
    // worker_ready_q[vm[j]->leader].push_back(vm[j]);
    // LOCK_RELEASE(worker_lock[vm[j]->leader]);
    }
  } 

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(start);
  auto epoch1 = start1_ms.time_since_epoch();
  //Start the TAODAG exeuction
  gotao_start();
  // //Finalize and claim resources back
  gotao_fini();
  end = std::chrono::system_clock::now();
  auto end1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(end);
  auto epoch1_end = end1_ms.time_since_epoch();
  std::chrono::duration<double> elapsed_seconds = end-start;
  // std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::ofstream timetask;
  timetask.open("data_process.sh", std::ios_base::app);
  timetask << "python Energy.py " << epoch1.count() << "\t" <<  epoch1_end.count() << "\n";
  timetask.close();
  std::cout << epoch1.count() << "\t" <<  epoch1_end.count() << ", execution time: " << elapsed_seconds.count() << "s. "<< std::endl;
  std::cout << "Result is " << D << std::endl;
  std::cout << "Done!\n";
  std::cout << "Total successful steals: " << tao_total_steals << std::endl;
}
