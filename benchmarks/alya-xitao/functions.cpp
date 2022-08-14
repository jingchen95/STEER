#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>
#include<vector>
#include <chrono>
#include "functions.h"
#include "../../include/xitao.h"
#define CHUNK 20000
//x = A*b
void spmv(int nrows,int* iA,int* jA, double* A, double* b, double* x)
{
    double sum;
    int istart,iend;

    for(int row = 0; row < nrows; row++){
        istart = iA[row];
        iend   = iA[row+1];
        sum = 0.0;
        for(int i = istart; i < iend; i++){
            sum += b[jA[i]] * A[i];
        }
        x[row] = sum;
    }
}

// diagonal preconditioner
void precond(int nrows,int* iA,int* jA, double* A, double* diag)
{
    int istart,iend;
    for(int row = 0; row < nrows; row++){
        istart = iA[row];
        iend   = iA[row+1];
        for(int i = istart; i < iend; i++){
            if( jA[i] == row )
                diag[row] = 1.0/A[i];
        }
    }
}


//y=x
void yequalsx(int nrows, double* y, double* x)
{
  for(int row = 0; row < nrows; row++){
    y[row] = x[row];
  }

}

//y=constant
void yequalsconst(int nrows, double* y, double constant)
{
  for(int row = 0; row < nrows; row++){
    y[row] = constant;
  }
}

//dot= x*y // dot product resulting in a scalar
double dotxy(int nrows, double* x, double* y, double &dot)
{

  dot=0.0;
  for(int row = 0; row < nrows; row++){
    dot += x[row] * y[row];
  }

  return dot;
}

//z= x*y // element wise multiplication resulting in a vector
void multxy(int nrows, double* z ,double* x, double*y)
{
  for(int row = 0; row < nrows; row++){
    z[row] = x[row] * y[row];
  }
}

// y= a*x + y // a is a scalar
void axpy(int nrows, double* y, double*x, double a)
{
  for(int row = 0; row < nrows; row++){
    y[row] = a*x[row] + y[row];
  }
}
// y= a*x + b*y 
void axpby(int nrows, double* y, double*x, double a, double b)
{
  for(int row = 0; row < nrows; row++){
    y[row] = a*x[row] + b*y[row];
  }
}

//solve conjugate gradient
void solve(int nrows, int* iA, int* jA, double* A, double* x, double *rhs, double* p0, double* r0, double *Ax, double* s, double* diag)
{
    double alpha,beta,rho1,rho0,temp1,temp2,tolmax;
    int itmax, it;
    tolmax = 1e-5;
    itmax  = 10000;

    yequalsx(nrows, p0, x);

    yequalsx(nrows, r0, rhs);

    spmv(nrows, iA, jA,  A,  p0, Ax );

    axpy(nrows, r0, Ax, -1.0);

    multxy(nrows, p0, diag, r0);

    dotxy(nrows, p0, r0, rho1);

    printf ("%d nrows\n", nrows);

    temp2 = 1.0;
    std::vector<double> partial_sum(gotao_nthreads);
    //gotao_init();
    gotao_init_hw(-1,-1,-1);
    int schedulerid = 1;
    gotao_init(schedulerid, 6, 0, 0);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(start);
    auto epoch1 = start1_ms.time_since_epoch();
    std::cout << "Alya starting time: " << epoch1.count() << std::endl;
    for(it = 0; ((tolmax <= sqrt(temp2)) || (it == 0)) && (it < itmax); it++)
    {
#ifdef DEBUG
      std::cout << "Iteration " << it << " Begins... \n";
#endif
    int row;       
    auto data_parallel_1 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      int istart = iA[row];
      int iend   = iA[row+1];
      double sum = 0.0;
      for(int i = istart; i < iend; i++){
        sum += p0[jA[i]] * A[i];
      }
      Ax[row] = sum;      
    );

    partial_sum.assign(gotao_nthreads, 0);
    auto data_parallel_2 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      partial_sum[__xitao_thread_id] += p0[row] * Ax[row];
    );

    temp1 = 0.0;
    for (auto& n : partial_sum) temp1 += n;
    alpha = rho1 / temp1;

    auto data_parallel_3 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      x[row] = alpha*p0[row] + x[row];
    );
    
    auto data_parallel_4 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      r0[row] = -alpha*Ax[row] + r0[row];
    );
   
    
    temp2 = 0.0;
    partial_sum.assign(gotao_nthreads, 0);
    auto data_parallel_5 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      partial_sum[__xitao_thread_id] += r0[row] * r0[row];
    );
   
    for (auto& n : partial_sum) temp2 += n;
    auto data_parallel_6 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      s[row] = diag[row] * r0[row];
    );      
   
    rho0 = rho1;
    rho1 = 0.0;
    partial_sum.assign(gotao_nthreads, 0);
    auto data_parallel_7 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      partial_sum[__xitao_thread_id] += r0[row] * s[row];
    );
   
    for (auto& n : partial_sum) rho1 += n;
    beta = rho1 / rho0;
    auto data_parallel_8 = __xitao_vec_multiparallel_region_sync(1, row, 0, nrows, xitao_vec_static, CHUNK,
      p0[row] = s[row] + beta * p0[row];
    );  
  #ifdef DEBUG
    std::cout << "Iteration " << it << " complete! \n";
  #endif
    }
    if( tolmax > sqrt(temp2) && it > 0 )
    {
      printf("Convergence at iteration %d tolerance %e \n",it, sqrt(temp2));
    }
    gotao_fini();
    end = std::chrono::system_clock::now();
    auto end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(end);
    auto epoch = end_ms.time_since_epoch();
    std::cout << "Alya ending time: " << epoch.count() << std::endl;
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed Time: " << elapsed_seconds.count() << std::endl;
#ifdef NUMTASKS
    for(int a = 1; a < 2; a = a*2){
      for(int b = 0; b < gotao_nthreads; b++){
          std::cout << "Thread " << b << " with width " << a << " completes " << num_task[a * gotao_nthreads + b] << " tasks.\n";
      }
    }
#endif
}

