#ifndef _CONFIG_H
#define _CONFIG_H

/* Open debug option*/
//#define DEBUG

/* 2021 Sep 28: paper II takes DVFS into account*/
// #define dvfs

/* 2021 Sep 28: Run same type of tasks with 2 different frequencies: currently 2.04GHz and 1.11GHz */
#define NUMFREQ 2

#define TRAIN_METHOD_1
#define Performance_Model_Time    // Only use time to calculate memory-boundness
//#define Performance_Model_Cycle // Use Cycles to calculate memory-boundness
#define FINE_GRAIN_THRESHOLD 0.0005
#define NUM_TRAIN_TASKS 6
#define START_D 0
#define START_A 2
#define NUM_AVAIL_FREQ 12
#define NUMSOCKETS 2

//#define EDP_TEST_  // Test minimize EDP per task

//#define FineStrategyTest
//#define FINE_GRAIN_THRESHOLD 0.0001

/* Performance Counters - calculate Memory-boundness */
// #define PERF_COUNTERS

/* Power Model for Hebbe in Synthetic DAG Benchmarks */
// #define Haswell
// #define NUMSOCKETS 2
// #define COREPERSOCKET 10

// Power Profiling Kernel tasks, especially for applications with multiple kernels
//#define PowerProfiling

/* Power Model for TX2 in Synthetic DAG Benchmarks */
#define TX2

/* Average same configuration */
#define AveCluster

/* Set different ERASE target (only one is allowed) */
//#define ERASE_target_perf
//#define ERASE_target_energy_method1
#define ERASE_target_energy_method2

/* Blancing EDP between Denver and A57 clusters */
// #define ERASE_target_edp_method1

/* Test if energy increase (cost) is lower than execution time reduction (benefit) */
//#define ERASE_target_edp_method2

/* Test if energy optimization can use the same method as EDP */
//#define ERASE_target_test

/* Check the PTT Accuracy between prediction and real time*/
//#define PTTaccuracy

/* Check the accuracy between the energy prediction and real energy */
//#define Energyaccuracy
/* If application has multiple kernels, can not use single ptt_full (sparseLU test)*/
//#define MultipleKernels


/* ERASE uses second energy efficient config - to compare the energy with most energy efficent one*/
//#define second_efficient

//#define CATA
//#define Hermes

// #define FREQLEVELS 2

// #define EAS_PTT_TRAIN 2
/* Frequency setting: 0=>MAX, 1=>MIN */
#define A57 0
#define DENVER 0

#define STEAL_ATTEMPTS 1

/* Idle sleep try */
#define IDLE_SLEEP 100
/*Sleep time bound settings (nanoseconds)*/
#define SLEEP_LOWERBOUND 1000000
#define SLEEP_UPPERBOUND 64000000

/* Schedulers with sleep for idle work stealing loop */
#define SLEEP

/* Random working stealing with sleep */
#define RWSS_SLEEP

/* Performance-oriented with sleep */
#define FCAS_SLEEP
#define CRI_COST
//#define CRI_PERF

// CATS Scheduler
//#define CATS

/* Energy aware scheduler */
#define EAS_SLEEP
#define EAS_NoCriticality

// #define ACCURACY_TEST

/* Enable or disable work stealing */
#define WORK_STEALING

/* Accumulte the total exec time this thread complete */
#define EXECTIME

/* Accumulte the number of task this thread complete */
#define NUMTASKS_MIX

/* Accumulte the PTT visiting time */
// #define OVERHEAD_PTT

// #define ONLYCRITICAL

// #define PARA_TEST

//#define NEED_BARRIER

//#define DynaDVFS

#define GOTAO_THREAD_BASE 0
#define GOTAO_NO_AFFINITY (1.0)
#define TASK_POOL 100
#define TAO_STA 1
#define XITAO_MAXTHREADS 6
#define L1_W   1
#define L2_W   2
#define L3_W   6
////#define L4_W   12
////#define L5_W   48
#define TOPOLOGY { L1_W, L2_W}
#define GOTAO_HW_CONTEXTS 1
#endif
