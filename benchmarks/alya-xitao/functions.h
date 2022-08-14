//x = A*b
void spmv(int nrows,int* iA,int* jA, double* A, double* b, double* x);
#if 0
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
#endif

// diagonal preconditioner
void precond(int nrows,int* iA,int* jA, double* A, double* diag);
#if 0
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
#endif


//y=x
void yequalsx(int nrows, double* y, double* x);
#if 0
{
	for(int row = 0; row < nrows; row++){
	  y[row] = x[row];
	}

}
#endif

//y=constant
void yequalsconst(int nrows, double* y, double constant);
#if 0
{
	for(int row = 0; row < nrows; row++){
	  y[row] = constant;
	}
}
#endif

//dot= x*y // dot product resulting in a scalar
double dotxy(int nrows, double* x, double* y, double &dot);
#if 0
{

	dot=0.0;
	for(int row = 0; row < nrows; row++){
	  dot += x[row] * y[row];
	}

	return dot;
}
#endif

//z= x*y // element wise multiplication resulting in a vector
void multxy(int nrows, double* z ,double* x, double*y);
#if 0
{
	for(int row = 0; row < nrows; row++){
	  z[row] = x[row] * y[row];
	}
}
#endif

// y= a*x + y // a is a scalar
void axpy(int nrows, double* y, double*x, double a);
#if 0
{
	for(int row = 0; row < nrows; row++){
	  y[row] = a*x[row] + y[row];
	}
}
#endif
// y= a*x + b*y 
void axpby(int nrows, double* y, double*x, double a, double b);
#if 0
{
	for(int row = 0; row < nrows; row++){
	  y[row] = a*x[row] + b*y[row];
	}
}
#endif

//solve conjugate gradient
void solve(int nrows, int* iA, int* jA, double* A, double* x, double *rhs, double* p0, double* r0, double *Ax, double* s, double* diag);
#if 0
{

    double alpha,beta,rho1,rho0,temp1,temp2,tolmax;
    int itmax;
    tolmax = 1e-5;
    itmax  = 10000;

    yequalsx(nrows, p0, x);

    yequalsx(nrows, r0, rhs);

    spmv(nrows, iA, jA,  A,  p0, Ax );

    axpy(nrows, r0, Ax, -1.0);

    multxy(nrows, p0, diag, r0);

    dotxy(nrows, p0, r0, rho1);

    for(int it = 0; it < itmax; it++)
    {
        spmv(nrows, iA, jA,  A,  p0, Ax );

        dotxy(nrows, p0, Ax, temp1);

        alpha = rho1 / temp1;

        axpy(nrows, x, p0, alpha);

        axpy(nrows, r0, Ax, -1.0*alpha);

        dotxy(nrows, r0, r0, temp2);

        if( tolmax > sqrt(temp2) && it > 0 )
        {
             printf("Convergence at iteration %d tolerance %e \n",it, sqrt(temp2));
             break;
        }
        multxy(nrows, s, diag, r0);
        
        rho0 = rho1;

        dotxy(nrows, r0, s, rho1);

        beta = rho1/rho0;

        axpby(nrows, p0, s, 1.0, beta);
    }
}
#endif
