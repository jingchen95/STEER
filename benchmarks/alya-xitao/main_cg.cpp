#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>
#include "functions.h"

#define MAXCHAR 100

using namespace std;
 
int main(int argc, char **argv){

    if(argc-1 != 1) {cout<<"Error in the input: ./solver.x C100K.csr  "<<endl;return 0;}
    char*  matname=argv[1];


 // start reading the matrix from file   
    cout<<"Reading matrix ..."<<endl;

    int nnz,num_cols,num_rows,blocks,threads;
    char filename[MAXCHAR];
    sprintf(filename,"%s",matname);

    FILE *fp;

    fp= fopen(filename,"r");
    if(fp == NULL)
    {
        cout<<"Error: matrix file "<<filename<<" not found"<<endl;
        return 0;
    }

    int sizes[5];

    for(int i=0;i<5;i++)
        fscanf(fp," %d",&sizes[i]);
    fscanf(fp," \n");

    nnz=sizes[0];
    num_cols=sizes[1];
    num_rows=sizes[2];

    double*   A=new double[nnz];
    int*      jA=new int[nnz];
    int*      iA= new int[num_rows+1];
    double*   x= new double[num_rows];
    double*   xsol= new double[num_rows];
    double*   b= new double[num_rows];


    for(int i=0;i<nnz;i++)
        fscanf(fp," %lf",&A[i]);
    fscanf(fp," \n");
    for(int i=0;i<nnz;i++)
        fscanf(fp," %d",&jA[i]);
    fscanf(fp," \n");
    for(int i=0;i<num_rows+1;i++)
        fscanf(fp," %d",&iA[i]);
    fscanf(fp," \n");

    fclose(fp);


    //Creates a manufactured solution stored in xsol
    for(int i=0;i<num_rows;i++)
	{
        xsol[i]= 0.1*sin(1.0*i) +0.01*sin(2.0*i) +0.001*sin(3.0*i)+ cos(0.1*i) + tan(0.1*i);
	    x[i]=0.0;
	}

    //Create a rhs which solution is xsol
    spmv(num_rows, iA, jA,  A,  xsol, b );


    //Create auxiliar variables
    double* p0   = new double[num_rows];
    double* r0   = new double[num_rows];
    double* Ax   = new double[num_rows];
    double* s    = new double[num_rows];
    double* diag = new double[num_rows];

    cout<<"Preconditioner setup ..."<<endl;

    //With a constant matrix the setup of the preconditioner is needed only once
    precond(num_rows, iA, jA, A, diag);
 

    cout<<"Solving ..."<<endl;
    solve(num_rows, iA, jA, A, x, b, p0, r0, Ax, s, diag);


    //Comparing the results with the manufactured solution
    double error=0.0;
    for(int k = 0; k < num_rows; k++)
    {
        if(fabs(xsol[k] - x[k]) > error)
            error =fabs(xsol[k] - x[k]);  
    }
    cout<<"Absolute error: "<<error<<endl;


}
