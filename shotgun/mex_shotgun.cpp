#include <math.h>
#include "mex.h"

#include "common.h"


// All matlab values are matrices, even scalars!
double get_scalar(const mxArray*prhs[], int idx) {
    double * d = mxGetPr(prhs[idx]);
    return d[0];
}


// Arguments:  A, y, lambda, algo (1=lasso,2=logreg), threshold, K, maxiter
void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    int n, d;
    if (nlhs != 1) {
        mexErrMsgTxt("You need one output argument");
    }
    
    if (nrhs < 9) {
        mexErrMsgTxt("You need nine input args (A, y, lambda, algo (1=lasso,2=logreg), threshold, K, maxiter, numthreads, verbose)!!");
    }
    
    
    if(mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]) || mxIsComplex(prhs[2])) {
        mexErrMsgTxt("Unfortunately Shotgun does not yet support complex arguments.");
    }
    if(mxIsLogical(prhs[0])) {
        mexErrMsgTxt("Logical matrices not supported.");
    }
    
    int algorithm = (int) get_scalar(prhs, 3);
    double threshold = get_scalar(prhs, 4);
    int K = (int) get_scalar(prhs, 5);
    int maxiter = (int) get_scalar(prhs, 6);
    int numthreads = (int) get_scalar(prhs, 7);
    int verbose = (int) get_scalar(prhs, 8);   
    
    omp_lock_t globallock;
    omp_init_lock(&globallock);
    
    if (numthreads>0) {
        #ifndef DISABLE_OMP
        omp_set_num_threads(numthreads);
        mexPrintf("OMP threads = %d\n", numthreads);
        #endif
    }
    
    if (!(algorithm == 1 || algorithm == 2)) {
        mexErrMsgTxt("Algorithm has to be either 1=lasso or 2=logreg.");
    }
    
    double * _lambda = mxGetPr(prhs[2]);
    
    d = mxGetN(prhs[0]);
    n = mxGetM(prhs[0]);
    mexPrintf("d = %d, n=%d, y-dim=%d\n", d,n, mxGetM(prhs[1]));
    
    if (n != mxGetM(prhs[1])) {
       mexErrMsgTxt("y-vector does not have correct dimension!");
    }
    
    // Initialize
    std::vector<sparse_array> *  A_cols = new std::vector<sparse_array>();
    std::vector<sparse_array> * A_rows = new std::vector<sparse_array>();
    A_cols->resize(d);
    A_rows->resize(n);
    const mxArray *A = prhs[0];
    
    if (mxIsSparse(prhs[0])) {
        // sparse
        mexPrintf("A was sparse\n");
        
         // Start reading the sparse array
         mwSize nz;
         mwIndex *Jc;
         double *pr;
         mwIndex *ir;
         Jc = mxGetJc (A);
         pr = mxGetPr (A);
         ir = mxGetIr (A);
         
         // Get number of non-zero elements
         nz = Jc[d];
         
         // Compressed column format, need to keep track
         int jcidx = 0;
         int jccur = Jc[1];
         int col = 0, row;
         double sum = 0, val;
         int i;
         for (i = 0; i < nz; i++)
         {
            if (jccur==0) {
                col++;
                jccur = Jc[col+1]-Jc[col];
                // Each row of Jc gives the number of non-zeros before
                // that column. Storage is column-major.
		if (jccur==0) continue;
            }
            val = pr[i];
            row = ir[i];
            jccur--;
            sum += val;
            
            (*A_cols)[col].add(row, val);
            (*A_rows)[row].add(col, val);
         }
         
         mexPrintf("Loaded A\n");
         
        
   } else {
       mexPrintf("A was dense.\n");
       double * Ad = mxGetPr(A);
       int i = 0;
       for(int col=0; col<d; col++) {
        for(int row=0; row<n; row++) {
           double val = Ad[i++];
           if (val != 0) {
               (*A_cols)[col].add(row, val);
               (*A_rows)[row].add(col, val);
           }
         }
        }
       mexPrintf("Loaded A\n");
   }
   
    std::vector< shotgun_data > probs;
    
    int nprobs = mxGetN(prhs[1]);
    mexPrintf("Number of problems: %d", nprobs);
    
     // Load y
    for(int ip=0; ip<nprobs; ip++) {
     shotgun_data prob;
     prob.lock = &globallock;
     prob.parallel = (nprobs == 1);
     prob.A_cols = A_cols;
     prob.A_rows = A_rows;
     prob.y.reserve(n);
     double * _y = mxGetPr(prhs[1]);
     for(int i=0; i<n; i++) prob.y.push_back(_y[i]);
    
     mexPrintf("Loaded Y\n");
    
     // Run shotgun!
     prob.nx = d;
     prob.ny = n;
     probs.push_back(prob);
    }


    bool all_zero = false;
     
     
if (nprobs > 1) {
#pragma omp parallel for    
    for(int ip=0; ip<nprobs; ip++) {
     shotgun_data &prob = probs[ip];
     if (algorithm == 1) {
         solveLasso(&prob, *_lambda, K, threshold, maxiter, verbose);
     } else if (algorithm == 2) {
         compute_logreg(&prob, *_lambda, threshold, maxiter, verbose, all_zero);
     }
    }
} else {
     shotgun_data &prob = probs[0];
     if (algorithm == 1) {
         solveLasso(&prob, *_lambda, K, threshold, maxiter, verbose);
     } else if (algorithm == 2) {
         compute_logreg(&prob, *_lambda, threshold, maxiter, verbose, all_zero);
     }
}
    
    for(int ip=0; ip<nprobs; ip++) {
         plhs[0] = mxCreateDoubleMatrix(d, nprobs, mxREAL); 
         double * r = mxGetPr(plhs[0]);
         int i = 0;
         for(int ip=0; ip<nprobs; ip++) {
              shotgun_data &prob = probs[ip];
             for(int j=0; j<d; j++) {
                r[i] = prob.x[i];
                i++;
            }
        }
    }
}
