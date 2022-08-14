#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "sparselu_taos.h"  
#include "sparselu.h"                                                      
#include "xitao.h"                                                                  
#include <assert.h>
#include <set>
#include <fstream>      
#include <string>
#include <sstream>

int NB = 10;
int BSIZE = 512;

typedef double ELEM;
vector<vector<ELEM*>> A;
vector<vector<AssemblyTask*>> dependency_matrix;

// Enable to output dot file. Recommended to use with NB < 16 
#define OUTPUT_DOT
#define FALSE (0)
#define TRUE (1)
#define TAO_WIDTH 1
using namespace xitao;
using namespace std;

float LU0::time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float FWD::time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float BDIV::time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float BMOD::time_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];

float LU0::power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float FWD::power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float BDIV::power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float BMOD::power_table[NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];

uint64_t LU0::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t FWD::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t BDIV::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t BMOD::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];

float LU0::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
float FWD::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
float BDIV::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
float BMOD::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];

bool LU0::time_table_state[NUMSOCKETS+1];
bool FWD::time_table_state[NUMSOCKETS+1];
bool BDIV::time_table_state[NUMSOCKETS+1];
bool BMOD::time_table_state[NUMSOCKETS+1];

bool LU0::best_config_state;
bool FWD::best_config_state;
bool BDIV::best_config_state;
bool BMOD::best_config_state;

bool LU0::enable_freq_change;
bool FWD::enable_freq_change;
bool BDIV::enable_freq_change;
bool BMOD::enable_freq_change;

int LU0::best_freqindex;
int FWD::best_freqindex;
int BDIV::best_freqindex;
int BMOD::best_freqindex;

int LU0::best_cluster;
int FWD::best_cluster;
int BDIV::best_cluster;
int BMOD::best_cluster;

int LU0::best_width;
int FWD::best_width;
int BDIV::best_width;
int BMOD::best_width;

std::atomic<int> LU0::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> FWD::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> BDIV::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> BMOD::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];

std::atomic<int> LU0::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> FWD::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> BDIV::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> BMOD::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];

extern int NUM_WIDTH_TASK[XITAO_MAXTHREADS];

string get_tao_name(AssemblyTask* task) 
{
#ifdef OUTPUT_DOT
   stringstream st; 
   stringstream st_address; 
   string address;
   BDIV* bdiv = dynamic_cast<BDIV*>(task);
   BMOD* bmod = dynamic_cast<BMOD*>(task);
   LU0* lu0   = dynamic_cast<LU0*>(task);
   FWD* fwd   = dynamic_cast<FWD*>(task);
   const int digits = 3;
   if(bdiv) {
      st_address << bdiv;
      address = st_address.str();
      address = address.substr(address.size() - digits, address.size());
      st << "bdiv_"<< address;
   } else if (bmod){
      st_address << bmod;
      address = st_address.str();
      address = address.substr(address.size() - digits, address.size());
      st << "bmod_"<< address;
   } else if (lu0) {
      st_address << lu0;
      address = st_address.str();
      address = address.substr(address.size() - digits, address.size());
      st << "lu0_"<< address;
   } else if (fwd) {
      st_address << fwd;
      address = st_address.str();
      address = address.substr(address.size() - digits, address.size());
      st << "fwd_"<< address;
   } else {
      std::cout << "error: unknown TAO type" << std::endl;
      exit(0);
   }  
   return st.str();
#else
   return "";
#endif
}

inline void init_dot_file(ofstream& file, const char* name) 
{
#ifdef OUTPUT_DOT
   file.open(name);
   file << "digraph G {" << endl;
#endif
}

inline void close_dot_file(ofstream& file) {
#ifdef OUTPUT_DOT  
   file << "}" << endl;
   file.close();
#endif
}

inline void write_edge(ofstream& file, string const src, int indxsrc, int indysrc, string const dst, int indxdst, int indydst) 
{
#ifdef OUTPUT_DOT  
   stringstream st;
   st << src <<"_" << indxsrc << "_" << indysrc;
   file << st.str() << " -> ";
   st.str("");
   st << dst <<"_" << indxdst << "_" << indydst << ";" << endl;
   file << st.str();
#endif
}

void generate_matrix_structure (int& bcount)
{
   int null_entry;
   for (int ii=0; ii<NB; ii++)
      for (int jj=0; jj<NB; jj++){
         null_entry=FALSE;
         if ((ii<jj) && (ii%3 !=0)) null_entry =TRUE;
         if ((ii>jj) && (jj%3 !=0)) null_entry =TRUE;
         if (ii%2==1) null_entry=TRUE;
         if (jj%2==1) null_entry=TRUE;
         if (ii==jj) null_entry=FALSE;
         if (null_entry==FALSE){
            bcount++;
            //A[ii][jj] = (ELEM *) new (BSIZE*BSIZE*sizeof(ELEM));
            A[ii][jj] = (ELEM *) new ELEM[BSIZE*BSIZE];
            if (A[ii][jj]==NULL) {
               printf("Out of memory\n");
               exit(1);
            }
         }
         else A[ii][jj] = NULL;
      }
}


void generate_matrix_values ()
{
   int init_val, i, j, ii, jj;
   ELEM *p;

   init_val = 1325;

   for (ii = 0; ii < NB; ii++) 
     for (jj = 0; jj < NB; jj++){
         p = A[ii][jj];
         if (p!=NULL)
            for (i = 0; i < BSIZE; i++) 
               for (j = 0; j < BSIZE; j++) {
                  init_val = (3125 * init_val) % 65536;
                  (*p) = (ELEM)((init_val - 32768.0) / 16384.0);
                  p++;
               }
     }
}

void print_structure(){
   ELEM *p;
   int sum = 0; 
   int ii, jj, i, j;
   printf ("Structure for matrix A\n");
   for (ii = 0; ii < NB; ii++) {
     for (jj = 0; jj < NB; jj++) {
        p = A[ii][jj];
        if (p!=NULL){
           {
              for(i =0; i < BSIZE; i++)
              {
                 for(j =0; j < BSIZE; j++)
                 {
                    printf("%+lg \n", p[i * BSIZE + j]);
                 }
                 printf("\n");
              }
           }
        }
     }
   }
}

vector<ELEM> read_structure()
{
   ELEM *p;
   int sum = 0; 
   int ii, jj, i, j;
   vector<ELEM> results; 
   for (ii = 0; ii < NB; ii++) {
     for (jj = 0; jj < NB; jj++) {
        p = A[ii][jj];
        if (p!=NULL)
        {
           {
              for(i =0; i < BSIZE; i++)
              {
                 for(j =0; j < BSIZE; j++)
                 {
                    results.push_back(p[i * BSIZE + j]);
                 }
              }
           }
        }
     }
   }
   return results;
}

ELEM *allocate_clean_block()
{
  int i,j;
  ELEM *p, *q;

  //p=(ELEM*)malloc(BSIZE*BSIZE*sizeof(ELEM));
  p= new ELEM[BSIZE*BSIZE];
  q=p;
  if (p!=NULL){
     for (i = 0; i < BSIZE; i++) 
        for (j = 0; j < BSIZE; j++){(*p)=(ELEM)0.0; p++;}
	
  }
  else printf ("OUT OF MEMORY!!!!!!!!!!!!!!!\n");
  return (q);
}


/* ************************************************************ */
/* Utility routine to measure time                              */
/* ************************************************************ */

double myusecond()
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return ((double) tv.tv_sec *4000000) + tv.tv_usec;
}

double mysecond()
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return ((double) tv.tv_sec) + ((double)tv.tv_usec*1e-6);
}

double gtod_ref_time_sec = 0.0;

float get_time()
{
    double t, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    // If this is the first invocation of through dclock(), then initialize the
    // "reference time" global variable to the seconds field of the tv struct.
    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double) tv.tv_sec;

    // Normalize the seconds field of the tv struct so that it is relative to the
    // "reference time" that was recorded during the first invocation of dclock().
    norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

    // Compute the number of seconds since the reference time.
    t = norm_sec + tv.tv_usec * 1.0e-6;

    return (float) t;
}


void sparselu_parallel(int schedulerid){
#if defined(Haswell)
   #define MAX_PACKAGES 16
   char event_names[MAX_PACKAGES][1][256];
   char filenames[MAX_PACKAGES][1][256];
   char basename[MAX_PACKAGES][256];
   char tempfile[256];
   long long before[MAX_PACKAGES][1];
   long long after[MAX_PACKAGES][1];
   int valid[MAX_PACKAGES][1];
   FILE *fff;
#endif
   float t_start,t_end;
   float time;
   int ii, jj, kk;
   int bcount = 0;

   ofstream file;
   init_dot_file(file, "sparselu.dot");

   ofstream outfile;
   outfile.open("DAG.txt", std::ios_base::app);

   generate_matrix_structure(bcount);
   generate_matrix_values();
   printf("Init OK Matrix is: %d (%d %d) # of blocks: %d memory is %ld MB\n", (NB*BSIZE), NB, BSIZE, bcount, bcount*sizeof(ELEM)/1024/1024);
   
   gotao_init_hw(-1,-1,-1);
   gotao_init(schedulerid, 3, 0, 0); // 3 kernels for training: FWD, BDIV, BMOD; too few LU0 tasks, only exist in the DAG beginning. 

   std::ofstream timetask;
   timetask.open("data_process.sh", std::ios_base::app);

   int prev_diag = -1;
   //Timing Start
   t_start=get_time();
   // for(int iter = 0; iter = 2; iter++){
   for (kk=0; kk<NB; kk++) {
      auto lu0 = new LU0(A[kk][kk], BSIZE, TAO_WIDTH); 
      lu0->criticality = 1;
      lu0->tasktype = 3;
      lu0->kernel_name = "LU0";
      outfile << "Creating LU0: " << lu0->taskid << endl;
      //lu0_objs.push_back(new LU0(A[kk][kk], BSIZE, TAO_WIDTH));

      //check if the task will get its input this from a previous task
      if(dependency_matrix[kk][kk]) {
        write_edge(file, get_tao_name(dependency_matrix[kk][kk]), kk, kk, get_tao_name(lu0), kk, kk);
        dependency_matrix[kk][kk]->make_edge(lu0);
      //   cout << "Creating LU0: " << lu0->taskid << endl;
      } 
      else {
         // gotao_push(lu0,0);
         int queue = rand() % gotao_nthreads; //LU0 tasks are randomly executed on Denver
         if(queue < START_A){ // Schedule to Denver
            lu0->width = pow(2, rand() % 2); // Width: 1 2
            lu0->leader = START_D + (rand() % (2/lu0->width)) * lu0->width;
         }else{ // Schedule to A57
            lu0->width = pow(2, rand() % 3); // Width: 1 2 4
            lu0->leader = START_A + (rand() % (4/lu0->width)) * lu0->width;
         }
         // outfile << "Random: Creating LU0 " << lu0->taskid << ", leader " << lu0->leader << ", width " << lu0->width << endl;
         LOCK_ACQUIRE(worker_lock[lu0->leader]);
         // worker_ready_q[lu0->leader].push_front(lu0);
         worker_ready_q[lu0->leader].push_back(lu0);
         LOCK_RELEASE(worker_lock[lu0->leader]);
      }
      dependency_matrix[kk][kk] = lu0;

      for (jj=kk+1; jj<NB; jj++)
         if (A[kk][jj] != NULL) {
            //fwd(A[kk][kk], A[kk][jj]);
            auto fwd =  new FWD(A[kk][kk], A[kk][jj], BSIZE, TAO_WIDTH);
            outfile << "Creating fwd: " << fwd->taskid << endl;
            fwd->criticality = 1;
            fwd->tasktype = 0;
            fwd->kernel_name = "FWD";
            if(dependency_matrix[kk][kk]) {
              write_edge(file, get_tao_name(dependency_matrix[kk][kk]), kk, kk, get_tao_name(fwd), kk, jj);
              dependency_matrix[kk][kk]->make_edge(fwd);
            }
            if(dependency_matrix[kk][jj]) {
              write_edge(file, get_tao_name(dependency_matrix[kk][jj]), kk, jj, get_tao_name(fwd), kk, jj);
              dependency_matrix[kk][jj]->make_edge(fwd);
            }
           dependency_matrix[kk][jj] = fwd;
         }

      for (ii=kk+1; ii<NB; ii++) 
         if (A[ii][kk] != NULL) {
            //bdiv (A[kk][kk], A[ii][kk]);
            auto bdiv = new BDIV(A[kk][kk], A[ii][kk], BSIZE, TAO_WIDTH);
            outfile << "Creating bdiv: " << bdiv->taskid << endl;
            bdiv->criticality = 1;
            bdiv->tasktype = 1;
            bdiv->kernel_name = "BDIV";
            if(dependency_matrix[kk][kk]) {
              write_edge(file, get_tao_name(dependency_matrix[kk][kk]), kk, kk, get_tao_name(bdiv), ii, kk);
              dependency_matrix[kk][kk]->make_edge(bdiv);
            }

            if(dependency_matrix[ii][kk]) {
              write_edge(file, get_tao_name(dependency_matrix[ii][kk]), ii, kk, get_tao_name(bdiv), ii, kk);
              dependency_matrix[ii][kk]->make_edge(bdiv);
            }
            dependency_matrix[ii][kk] = bdiv;
         }

      for (ii=kk+1; ii<NB; ii++) {
         if (A[ii][kk] != NULL) {
            for (jj=kk+1; jj<NB; jj++) {
               if (A[kk][jj] != NULL) {
                  if (A[ii][jj]==NULL)
                  {
                     A[ii][jj]=allocate_clean_block();
                  }

                  //bmod(A[ii][kk], A[kk][jj], A[ii][jj]);
                  auto bmod = new BMOD(A[ii][kk], A[kk][jj], A[ii][jj], BSIZE, TAO_WIDTH);
                  outfile << "Creating bmod: " << bmod->taskid << endl;
                  if(ii == jj){
                     bmod->criticality = 1;
                  }
                  bmod->tasktype = 2;
                  bmod->kernel_name = "BMOD";
                  if(dependency_matrix[ii][kk]) {
                      write_edge(file, get_tao_name(dependency_matrix[ii][kk]), ii, kk, get_tao_name(bmod), ii, jj);
                      dependency_matrix[ii][kk]->make_edge(bmod);
                  }

                  if(dependency_matrix[kk][jj]) {
                      write_edge(file, get_tao_name(dependency_matrix[kk][jj]), kk, jj, get_tao_name(bmod), ii, jj);
                      dependency_matrix[kk][jj]->make_edge(bmod);
                  }

                  if(dependency_matrix[ii][jj]) {
                    write_edge(file, get_tao_name(dependency_matrix[ii][jj]), ii, jj, get_tao_name(bmod), ii, jj);
                    dependency_matrix[ii][jj]->make_edge(bmod);
                  }
                  dependency_matrix[ii][jj] = bmod;
               }
            }
         }
      }
   }
   // }
   //Timing Stop
   t_end=get_time();
   time = t_end-t_start;
   printf("Building DAG: %11.4f sec\n", time);
   close_dot_file(file);
   
#if defined(Haswell)
   int iii,jjj = 0; 
   for(jjj=0;jjj<NUMSOCKETS;jjj++) {
      sprintf(basename[jjj],"/sys/class/powercap/intel-rapl/intel-rapl:%d",jjj);
      sprintf(tempfile,"%s/name",basename[jjj]);
      fff=fopen(tempfile,"r");
      if (fff==NULL) {
         fprintf(stderr,"\tCould not open %s\n",tempfile);
         return;
      }
      fscanf(fff,"%s",event_names[jjj][iii]);
      valid[jjj][iii]=1;
      fclose(fff);
      sprintf(filenames[jjj][iii],"%s/energy_uj",basename[jjj]);
   }
   /* Gather before values */
   for(jjj=0;jjj<NUMSOCKETS;jjj++) {
      if (valid[jjj][iii]) {
      fff=fopen(filenames[jjj][iii],"r");
      if (fff==NULL) {
         fprintf(stderr,"\tError opening %s!\n",filenames[jjj][iii]);
      }
      else {
         fscanf(fff,"%lld",&before[jjj][iii]);
         fclose(fff);
      }
      }
   }
#endif
   //Timing Start
   // t_start=get_time();
   std::chrono::time_point<std::chrono::system_clock> start, end;
   start = std::chrono::system_clock::now();
   auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(start);
   auto epoch1 = start1_ms.time_since_epoch();
   //gotao_push(lu0_objs[0]);
   //Start the TAODAG exeuction
   gotao_start();
   //Finalize and claim resources back
   gotao_fini();
   //Timing Stop
   // t_end=get_time();
   end = std::chrono::system_clock::now();
   auto end1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(end);
   auto epoch1_end = end1_ms.time_since_epoch();
   timetask << "python Energy.py " << epoch1.count() << "\t" <<  epoch1_end.count() << "\n";
   // timetask.close();
#if defined(Haswell)
   /* Gather after values */
   for(jjj=0;jjj < NUMSOCKETS;jjj++) {
      if (valid[jjj][iii]) {
        fff=fopen(filenames[jjj][iii],"r");
        if (fff==NULL) {
          fprintf(stderr,"\tError opening %s!\n",filenames[jjj][iii]);
        }
        else {
          fscanf(fff,"%lld",&after[jjj][iii]);
          fclose(fff);
        }
      }
   }
#endif
   // time = t_end-t_start;
   std::chrono::duration<double> elapsed_seconds = end-start;
   std::cout << epoch1.count() << "\t" <<  epoch1_end.count() << "," << elapsed_seconds.count() << std::endl;
   // printf("Matrix is: %d (%d %d) memory is %ld MB time to compute in parallel: %11.4f sec\n", (NB*BSIZE), NB, BSIZE, (NB*BSIZE)*(NB*BSIZE)*sizeof(ELEM)/1024/1024, time);
   // print_structure();
#if defined(Haswell)
   std::cout << "Energy: " << (((double)after[0][0]-(double)before[0][0])+((double)after[1][0]-(double)before[1][0]))/1000000.0 << "\n";
#if (defined NUMTASKS)
   std::cout << NUM_WIDTH_TASK[1] << " tasks complete with width 1. \n";
   std::cout << NUM_WIDTH_TASK[2]/2 << " tasks complete with width 2. \n";
   std::cout << NUM_WIDTH_TASK[5]/5 << " tasks complete with width 5. \n";
   std::cout << NUM_WIDTH_TASK[10]/10 << " tasks complete with width 10. \n\n\n";
#endif
#endif
}

void sparselu_serial() {
  float t_start,t_end;
  float time;
  int ii, jj, kk;
  int bcount = 0;
  generate_matrix_structure(bcount);
  generate_matrix_values();
  //print_structure();
  printf("Init OK Matrix is: %d (%d %d) # of blocks: %d memory is %ld MB\n", (NB*BSIZE), NB, BSIZE, bcount, bcount*sizeof(ELEM)/1024/1024);

  //Timing Start
  t_start=get_time();

  for (kk=0; kk<NB; kk++) {

     lu0(A[kk][kk], BSIZE);

     for (jj=kk+1; jj<NB; jj++)
        if (A[kk][jj] != NULL)
           fwd(A[kk][kk], A[kk][jj], BSIZE);

     for (ii=kk+1; ii<NB; ii++) 
        if (A[ii][kk] != NULL)
           bdiv(A[kk][kk], A[ii][kk], BSIZE);

     for (ii=kk+1; ii<NB; ii++) {
        if (A[ii][kk] != NULL) {
           for (jj=kk+1; jj<NB; jj++) {
              if (A[kk][jj] != NULL) {
                 if (A[ii][jj]==NULL)
                 {
                    A[ii][jj]=allocate_clean_block();
                 }

                 bmod(A[ii][kk], A[kk][jj], A[ii][jj], BSIZE);
              }
           }
        }
     }
  }
  //Timing Stop
  t_end=get_time();
  time = t_end-t_start;
  printf("Matrix is: %d (%d %d) memory is %ld MB time to compute in serial: %11.4f sec\n", (NB*BSIZE), NB, BSIZE, (NB*BSIZE)*(NB*BSIZE)*sizeof(ELEM)/1024/1024, time);
}

/* ************************************************************ */
/* main               */
/* ************************************************************ */

int main(int argc, char** argv) {
  if(argc < 4) {
    printf("Usage: ./sparselu SchedulerID BLOCKS BLOCKSIZE\n");
    exit(0);
  }
  int schedulerid = atoi(argv[1]);
  NB = atoi(argv[2]);
  BSIZE = atoi(argv[3]);
  A.resize(NB);
  dependency_matrix.resize(NB);
  for(int i = 0; i < NB; ++i) {
    A[i].resize(NB, NULL);
    dependency_matrix[i].resize(NB, NULL);
  }
  const char *scheduler[] = { "PerformanceBased", "EnergyAware", "EDPAware", "RWSS"/* etc */ };
  std::cout << "---------------------- Test Application - Sparse LU ---------------------\n";
  std::cout << "--------- You choose " << scheduler[schedulerid] << " scheduler ---------\n";
  sparselu_parallel(schedulerid);
}

