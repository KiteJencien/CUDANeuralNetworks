//
// Created by DanielSun on 8/30/2021.
//
#include<cstdio>
#include<cuda.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include<cmath>
#include<numbers>
#include "LogUtils.cuh"


#ifndef NETWORKS_MATRIXOPERATIONS_CUH
#define NETWORKS_MATRIXOPERATIONS_CUH


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

    //call the copy of mat
    static void callMatCopy( Matrix2d *src, Matrix2d *dist);

    //mem allocation for elements
    static void callAllocElement(Matrix2d *mat1, int row, int col);

    //initialization of empty Matrix
    static void callAllocRandom(Matrix2d *mat1);
    static void callAllocZero(Matrix2d *mat1);

};

#endif //NETWORKS_MATRIXOPERATIONS_CUH
