//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_MATRIXOPERATIONS_CUH
#define CUDANETWORKS_MATRIXOPERATIONS_CUH

#include<cuda.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include "LogUtils.cuh"

class MatrixOperations {
public:
    //Matrix define : M(row, col) = *(M.elements + row * M.colcount + col)
    struct Matrix2d {
        int colcount;
        int rowcount;
        float *elements;   //content
    };

    //get the element at the particular location
    static __device__ float getElement2D(Matrix2d *source, int row, int col);

    //set the element at location
    static __device__ void setElement2D(Matrix2d *source, int row, int col, float value);
    static void callSetElement(Matrix2d *mat1, int row, int col, float value);

    //call cross product method
    static void callCrossProduct2D( Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static void callConstantProduct2D( Matrix2d *mat1, float k);

    //call addition method, result will be stored in mat1
    static void callAddition( Matrix2d *mat1, Matrix2d *mat2);
    static void callAddition( Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);

    static void callSubtraction( Matrix2d *mat1, Matrix2d *mat2);
    static void callSubtraction( Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);

    static void callExponential( Matrix2d *mat1, float value);
    static void callExponential( Matrix2d *mat1, float value, Matrix2d *result);

    //call sigmoid method, result will be stored in mat1
    static void callSigmoidActivation( Matrix2d *mat1);
    static void callSigmoidActivation( Matrix2d *mat1, Matrix2d *result);

    //call sigmoid derivative method:
    static void callSigmoidDerivative( Matrix2d *mat1);

    //call hadmardProduct method:
    static void callHardmardProduct( Matrix2d *mat1, Matrix2d *mat2);

    //call transit method:
    static void callTransit2D( Matrix2d *mat1, Matrix2d *result);

    //flatten the matrix:
    static void callFlatten(Matrix2d *mat1, Matrix2d *mat2);

    //call the copy of mat
    static void callMatCopyD2D( Matrix2d *src, Matrix2d *dist);
    static void callMatCopyH2D( Matrix2d *src, Matrix2d *dist);
    static void callMatCopyD2H( Matrix2d *src, Matrix2d *dist);

    //mem allocation for elements
    static void callAllocElementD(Matrix2d *mat1, int row, int col);
    static void callAllocElementH(Matrix2d *mat1, int row, int col);
    static void callFreeElement(Matrix2d *mat1);

    //initialization of empty Matrix
    static void callAllocRandom(Matrix2d *mat1);
    static void callAllocZero(Matrix2d *mat1);


    //Debug tools
    static void inspect(Matrix2d* mat1);

};


#endif //CUDANETWORKS_MATRIXOPERATIONS_CUH
