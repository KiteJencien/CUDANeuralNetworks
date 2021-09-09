//
// Created by DanielSun on 8/30/2021.
//
#include<cstdio>
#include<cuda.h>
#include<cuda_runtime.h>
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
    static void callCrossProduct2D(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static void callConstantProduct2D(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, float k);

    //call addition method, result will be stored in mat1
    static void callAddition(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *mat2);
    static void callAddition(dim3 gridSize, dim3 blockSzie, Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);

    static void callSubtraction(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *mat2);
    static void callSubtraction(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);

    static void callExponential(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, float value);
    static void callExponential(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, float value, Matrix2d *result);

    //call sigmoid method, result will be stored in mat1
    static void callSigmoidActivation(dim3 gridSize, dim3 blockSize, Matrix2d *mat1);
    static void callSigmoidActivation(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *result);

    //call sigmoid derivative method:
    static void callSigmoidDerivative(dim3 gridSize, dim3 blockSize, Matrix2d *mat1);

    //call hadmardProduct method:
    static void callHardmardProduct(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *mat2);

    //call transit method:
    static void callTransit2D(dim3 gridSize, dim3 blockSize, Matrix2d *mat1, Matrix2d *result);

    //call the copy of mat
    static void callMatCopy(dim3 gridSize, dim3 blockSize, Matrix2d *src, Matrix2d *dist);

};

#endif //NETWORKS_MATRIXOPERATIONS_CUH
