/***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <time.h>

#include "cuda.h"
#include "device_functions.h"
#include "Reg_TVL1_Newton_kernels.cu"

#define BLOCKSIZE 512
#define MAX_GRID_SIZE 65535
#define NUMTHREADS 512

dim3 GetGrid(int size){
    size = (size-1) / NUMTHREADS + 1;
    dim3 grid( size, 1, 1 );
    if( grid.x > MAX_GRID_SIZE ) grid.x = grid.y = (int) sqrt( (double)(size-1) ) + 1;
    else if( grid.y > MAX_GRID_SIZE ) grid.x = grid.y = grid.z = (int) pow( (double)(size-1), (double)1.0/3.0 ) + 1;
    return grid;
}

extern void mexFunction(int iNbOut, mxArray *pmxOut[],
        int iNbIn, const mxArray *pmxIn[]){
    
    /* iNbOut: number of outputs */
    /* pmxOut: array of pointers to output arguments */
    
    /* iNbIn: number of inputs
    /* pmxIn: array of pointers to input arguments */
    
    /*  host arrays and variables */
    float   *h_ux, *h_uy, *h_uz, *h_cvg, *h_Ux, *h_Uy, *h_Uz;
    float   *h_VecParameters,*h_Gx, *h_Gy, *h_Gz, *h_Gf, *h_Gt;
    float   *h_bx1, *h_bx2, *h_bx3, *h_by1, *h_by2, *h_by3, *h_bz1, *h_bz2, *h_bz3;
    float   *h_q, *h_gkx, *h_gky, *h_gkz, *tt, *h_dvx, *h_dvy, *h_dvz;
    float   fError, cc, steps, fPenalty, fps;
    /*
    int     *punum, iNy, iNx, iNz, iNdim, iDim[3], iNI;
    int     maxIter, SZF, iDev;
    */
    int     *punum, iNy, iNx, iNz, iNdim, iDim[3], maxIter;
    
    
    cudaSetDevice(1);
    
    /* Timing */
    cudaEvent_t start, stop;
    float time;
    
    /*  device arrays */
    float   *d_bx1, *d_by1, *d_bz1, *d_bx2, *d_by2, *d_bz2, *d_bx3, *d_by3, *d_bz3, *d_dvx;
    float   *d_q, *d_dvy, *d_dvz, *d_gkx, *d_gky, *d_gkz, *d_ux, *d_uy, *d_uz;
    float   *d_Ux, *d_Uy, *d_Uz, *d_Gx, *d_Gy, *d_Gz, *d_Gf, *d_Gt, *h_FPS, *d_FPS;
    
    
    /* CUDA event-based timer start */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    
    /* input interface with matlab arrays */
    h_VecParameters = (float *)mxGetData(pmxIn[0]); /* Vector of parameters */
    h_Ux = (float *)mxGetData(pmxIn[1]);
    h_Uy = (float *)mxGetData(pmxIn[2]);
    h_Uz = (float *)mxGetData(pmxIn[3]);
    h_Gx = (float *)mxGetData(pmxIn[4]);
    h_Gy = (float *)mxGetData(pmxIn[5]);
    h_Gz = (float *)mxGetData(pmxIn[6]);
    h_Gt = (float *)mxGetData(pmxIn[7]);
    h_Gf = (float *)mxGetData(pmxIn[8]);
    
    
    /* dimensions */
    iNy = (int) h_VecParameters[0];
    iNx = (int) h_VecParameters[1];
    iNz = (int) h_VecParameters[2];
    
    unsigned int imageSize = iNx*iNy*iNz;
    
    /* parameters */
    maxIter = (int) h_VecParameters[3]; /* total number of iterations */
    fError = (float) h_VecParameters[4]; /* error criterion */
    cc = (float) h_VecParameters[5]; /* cc for ALM */
    steps = (float) h_VecParameters[6]; /* steps for each iteration */
    fPenalty = (float) h_VecParameters[7];
    
    /* output interface with matlab */
    /* ux */
    iNdim = 3;
    iDim[0] = iNy;
    iDim[1] = iNx;
    iDim[2] = iNz;
    
    pmxOut[0] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    h_ux = (float*)mxGetData(pmxOut[0]);
    
    /* uy */
    iNdim = 3;
    iDim[0] = iNy;
    iDim[1] = iNx;
    iDim[2] = iNz;
    
    
    pmxOut[1] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    h_uy = (float*)mxGetData(pmxOut[1]);
    
    /* uz */
    iNdim = 3;
    iDim[0] = iNy;
    iDim[1] = iNx;
    iDim[2] = iNz;
    
    pmxOut[2] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    h_uz = (float*)mxGetData(pmxOut[2]);
    
    /* convergence rate */
    iNdim = 2;
    iDim[0] = 1;
    iDim[1] = maxIter;
    pmxOut[3] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    h_cvg = (float*)mxGetData(pmxOut[3]);
    
    /* number of iterations */
    iNdim = 2;
    iDim[0] = 1;
    iDim[1] = 1;
    pmxOut[4] = mxCreateNumericArray(iNdim,(const int*)iDim,mxUINT16_CLASS,mxREAL);
    punum = (int*)mxGetData(pmxOut[4]);
    
    /* computation time */
    iNdim = 2;
    iDim[0] = 1;
    iDim[1] = 1;
    pmxOut[5] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    tt = (float*)mxGetData(pmxOut[5]);
    
    /* allocate host memory */
    /* bx1, bx2, bx3 */
    h_bx1 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_bx2 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_bx3 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_bx1 || !h_bx2 || !h_bx3) mexPrintf("calloc: Memory allocation failure\n");
    
    /* by1, by2, by3 */
    h_by1 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_by2 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_by3 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_by1 || !h_by2 || !h_by3) mexPrintf("calloc: Memory allocation failure\n");
    
    /* bz1, bz2, bz3 */
    h_bz1 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_bz2 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_bz3 = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_bz1 || !h_bz2 || !h_bz3) mexPrintf("calloc: Memory allocation failure\n");
    
    /* q */
    h_q = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_q) mexPrintf("calloc: Memory allocation failure\n");
    
    /* gk */
    h_gkx = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_gky = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_gkz = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_gkx || !h_gky || !h_gkz) mexPrintf("calloc: Memory allocation failure\n");
    
    /* div */
    h_dvx = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_dvy = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    h_dvz = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_dvx || !h_dvy || !h_dvz ) mexPrintf("calloc: Memory allocation failure\n");
    
    /* h_FPS */
    h_FPS = (float *) calloc( (unsigned)imageSize, sizeof(float) );
    if (!h_FPS) mexPrintf("calloc: Memory allocation failure\n");
    
    
    
    /* device memory allocation */
    cudaMalloc( (void**) &d_bx1, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_bx2, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_bx3, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_by1, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_by2, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_by3, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_bz1, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_bz2, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_bz3, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_gkx, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_gky, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_gkz, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_dvx, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_dvy, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_dvz, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_q,  sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_ux, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_uy, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_uz, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Ux, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Uy, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Uz, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Gx, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Gy, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Gz, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Gt, sizeof(float)*(unsigned)imageSize);
    cudaMalloc( (void**) &d_Gf, sizeof(float)*(unsigned)imageSize);
    
    cudaMalloc( (void**) &d_FPS, sizeof(float)*(unsigned)imageSize);
    
    /* copy arrays from host to device */
    cudaMemcpy( d_bx1, h_bx1, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_bx2, h_bx2, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_bx3, h_bx3, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_by1, h_by1, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_by2, h_by2, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_by3, h_by3, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_bz1, h_bz1, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_bz2, h_bz2, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_bz3, h_bz3, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_gkx, h_gkx, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_gky, h_gky, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_gkz, h_gkz, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_dvx, h_dvx, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_dvy, h_dvy, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_dvz, h_dvz, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_q,  h_q,  sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_ux, h_ux, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_uy, h_uy, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_uz ,h_uz, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Ux, h_Ux, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Uy, h_Uy, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Uz, h_Uz, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Gx, h_Gx, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Gy, h_Gy, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Gz, h_Gz, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Gt, h_Gt, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_Gf, h_Gf, sizeof(float)*(unsigned)imageSize, cudaMemcpyHostToDevice);
    
    /* run optimization */
    
    /* iNI = 0; */
    dim3 threads(BLOCKSIZE,1,1);
    dim3 grid = GetGrid(imageSize);
    
    for( int i = 0; i < maxIter; i++){
        
        /* update p */
        krnl_1<<<grid, threads>>>(d_dvx, d_dvy, d_dvz, d_ux, d_uy, d_uz,
                d_gkx, d_gky, d_gkz, d_Gx, d_Gy, d_Gz,
                d_Gt, d_Gf, d_q, d_Ux, d_Uy, d_Uz,
                cc, iNx, iNy, iNz);

        /* update px, py, pz */
        krnl_23z<<<grid, threads>>>(d_bx1, d_by1, d_bz1, 
                d_bx2, d_by2, d_bz2,
                d_bx3, d_by3, d_bz3,
                d_gkx, d_gky, d_gkz,
                steps, iNx, iNy, iNz);
        /*
//         krnl_2<<<grid, threads>>>(d_bx1, d_by1, d_bz1, d_gkx, d_gky, d_gkz,
//                 steps, iNx, iNy, iNz);
// 
//         krnl_3<<<grid, threads>>>(d_bx2, d_by2, d_bz2, d_gkx, d_gky, d_gkz,
//                 steps, iNx, iNy, iNz);
//         
//         krnl_z<<<grid, threads>>>(d_bx3, d_by3, d_bz3, d_gkx, d_gky, d_gkz,
//                 steps, iNx, iNy, iNz);
        */
        
        /* projection step */
        krnl_4<<<grid, threads>>>(d_bx1, d_bx2, d_bx3, d_by1, d_by2, d_by3,
                d_bz1, d_bz2, d_bz3, d_gkx, d_gky, d_gkz,
                fPenalty, iNx, iNy, iNz);
        
        krnl_56zp<<<grid, threads>>>(d_bx1, d_by1, d_bz1,
                d_bx2, d_by2, d_bz2,
                d_bx3, d_by3, d_bz3,
                d_gkx, d_gky, d_gkz,
                iNx, iNy, iNz);
        /*
//         krnl_5<<<grid, threads>>>(d_bx1, d_by1, d_bz1, d_gkx, d_gky, d_gkz,
//                 iNx, iNy, iNz);
//         
//         krnl_6<<<grid, threads>>>(d_bx2, d_by2, d_bz2, d_gkx, d_gky, d_gkz,
//                 iNx, iNy, iNz);
//         
//         krnl_zp<<<grid, threads>>>(d_bx3, d_by3, d_bz3, d_gkx, d_gky, d_gkz,
//                 iNx, iNy, iNz);
        */
        krnl_7<<<grid, threads>>>(d_bx1, d_bx2, d_bx3, d_by1, d_by2, d_by3,
                d_bz1, d_bz2, d_bz3, d_dvx, d_dvy, d_dvz, d_Gx, d_Gy, d_Gz,
                d_q, d_ux, d_uy, d_uz, d_FPS,
                cc, iNx, iNy, iNz);
        
        /* compute convergence */
        cudaMemcpy( h_FPS, d_FPS, sizeof(float)*unsigned(imageSize), cudaMemcpyDeviceToHost);
        
        fps = 0;
        for (int j=0; j< imageSize; j++){
            fps += abs(h_FPS[j]);
        }
        
        h_cvg[i] = fps / (float)imageSize;
        
        if (h_cvg[i] <= fError){
            break; 
        }
        
        /*mexPrintf("cvg: %f\n",h_cvg[i]); */
        
        punum[0] = i+1;
        
    }
    
    /* copy arrays from device to host */
    cudaMemcpy( h_ux, d_ux, sizeof(float)*(unsigned)(imageSize), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_uy, d_uy, sizeof(float)*(unsigned)(imageSize), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_uz, d_uz, sizeof(float)*(unsigned)(imageSize), cudaMemcpyDeviceToHost);
    
    mexPrintf("number of iterations = %i\n",punum[0]);
    
    
    /* Free memory */
    free( (float *) h_bx1 );
    free( (float *) h_bx2 );
    free( (float *) h_bx3 );
    free( (float *) h_by1 );
    free( (float *) h_by2 );
    free( (float *) h_by3 );
    free( (float *) h_bz1 );
    free( (float *) h_bz2 );
    free( (float *) h_bz3 );
    free( (float *) h_gkx );
    free( (float *) h_gky );
    free( (float *) h_gkz );
    free( (float *) h_dvx );
    free( (float *) h_dvy );
    free( (float *) h_dvz );
    free( (float *) h_q );
    
    free( (float *) h_FPS );
    
    /*    Free GPU Memory */
    cudaFree(d_bx1);
    cudaFree(d_bx2);
    cudaFree(d_bx3);
    cudaFree(d_by1);
    cudaFree(d_by2);
    cudaFree(d_by3);
    cudaFree(d_bz1);
    cudaFree(d_bz2);
    cudaFree(d_bz3);
    cudaFree(d_gkx);
    cudaFree(d_gky);
    cudaFree(d_gkz);
    cudaFree(d_dvx);
    cudaFree(d_dvy);
    cudaFree(d_dvz);
    
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
    cudaFree(d_Ux);
    cudaFree(d_Uy);
    cudaFree(d_Uz);
    cudaFree(d_Gx);
    cudaFree(d_Gy);
    cudaFree(d_Gz);
    cudaFree(d_Gt);
    cudaFree(d_Gf);
    cudaFree(d_q);
    cudaFree(d_FPS);
    
    /* CUDA event-based timer */
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    
    
    tt[0] = time;
    
    mexPrintf("\nComputational Time for Dual Optimization = %.4f sec\n \n",tt[0]/1000000);
    
    
}
