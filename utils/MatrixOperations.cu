//
// Created by DanielSun on 8/30/2021.
//

#include "MatrixOperations.cuh"

using namespace std;
//global functions: operations

//cross product
__global__ void crossProduct2D(MatrixOperations::Matrix2d *mat1, MatrixOperations::Matrix2d *mat2,
                               MatrixOperations::Matrix2d *resultBuffer) {
    float currentValue = 0.0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < mat1->colcount; ++i) {
        currentValue += MatrixOperations::getElement2D(mat1, row, i) *
                        MatrixOperations::getElement2D(mat2, i, col);
    }
    MatrixOperations::setElement2D(resultBuffer, row, col, currentValue);
}

__global__ void constantProduct2D(MatrixOperations::Matrix2d *mat1, float k){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = k*MatrixOperations::getElement2D(mat1, row, col);
    MatrixOperations::setElement2D(mat1, row, col, currentValue);
}

__global__ void transit2D(MatrixOperations::Matrix2d *input, MatrixOperations::Matrix2d *result){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(input, row, col);
    MatrixOperations::setElement2D(result, col, row, currentValue);
}


__global__ void addition2D(MatrixOperations::Matrix2d *input,MatrixOperations::Matrix2d *mat2){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(input, row, col);
    currentValue += MatrixOperations::getElement2D(mat2, row, col);
    MatrixOperations::setElement2D(input, row, col, currentValue);
}

__global__ void addition2D(MatrixOperations::Matrix2d *mat1, MatrixOperations::Matrix2d *mat2,
                           MatrixOperations::Matrix2d *output){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(mat1, row, col);
    currentValue += MatrixOperations::getElement2D(mat2, row, col);
    MatrixOperations::setElement2D(output, row, col, currentValue);
}

__global__ void subtraction2D(MatrixOperations::Matrix2d *mat1, MatrixOperations::Matrix2d *mat2){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(mat1, row, col);
    currentValue -= MatrixOperations::getElement2D(mat2, row, col);
    MatrixOperations::setElement2D(mat1, row, col, currentValue);
}
__global__ void subtraction2D(MatrixOperations::Matrix2d *mat1, MatrixOperations::Matrix2d *mat2,
                              MatrixOperations::Matrix2d *result){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(mat1, row, col);
    currentValue -= MatrixOperations::getElement2D(mat2, row, col);
    MatrixOperations::setElement2D(mat1, row, col, currentValue);
}

__global__ void exponential2D(MatrixOperations::Matrix2d *mat1, float value){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(mat1, row, col);
    currentValue = pow(currentValue, value);
    MatrixOperations::setElement2D(mat1, row, col, currentValue);
}

__global__ void exponential2D(MatrixOperations::Matrix2d *mat1, float value, MatrixOperations::Matrix2d *result){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(mat1, row, col);
    currentValue = pow(currentValue, value);
    MatrixOperations::setElement2D(result, row, col, currentValue);
}

__global__ void hadmardProduct(MatrixOperations::Matrix2d *mat1, MatrixOperations::Matrix2d *mat2){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(mat1, row, col);
    currentValue *= MatrixOperations::getElement2D(mat2,row,col);
    MatrixOperations::setElement2D(mat1, row, col, currentValue);
}

__global__ void derivativeSigmoid(MatrixOperations::Matrix2d *input){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(input, row, col);
    currentValue =  (1.0F/(1.0F + exp(-currentValue))) * (1.0F-(1.0F/(1.0F + exp(-currentValue))));
    MatrixOperations::setElement2D(input, row, col, currentValue);
}

__global__ void activationSigmoid(MatrixOperations::Matrix2d *input){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(input, row, col);
    currentValue = 1.0F/(1.0F + exp(-currentValue));  //sigmoid function
    MatrixOperations::setElement2D(input, row, col, currentValue);
}

__global__ void activationSigmoid(MatrixOperations::Matrix2d *input, MatrixOperations::Matrix2d *result){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(input, row, col);
    currentValue = 1.0F/(1.0F + exp(-currentValue));  //sigmoid function
    MatrixOperations::setElement2D(result, row, col, currentValue);
}

__global__ void MatCpy(MatrixOperations::Matrix2d *source, MatrixOperations::Matrix2d *dist){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float currentValue = MatrixOperations::getElement2D(source, row, col);
    MatrixOperations::setElement2D(dist, row, col, currentValue);
}

__device__ float MatrixOperations::getElement2D(MatrixOperations::Matrix2d *source, int row, int col) {
    return source->elements[row * source->colcount + col];
}

__device__ void MatrixOperations::setElement2D(MatrixOperations::Matrix2d *source, int row, int col, float value) {
    source->elements[row * source->colcount + col] = value;
}

void MatrixOperations::callCrossProduct2D(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                          MatrixOperations::Matrix2d *mat2, MatrixOperations::Matrix2d *result) {
    //condition
    if (mat1->colcount != mat2->rowcount || mat1->colcount != result -> colcount
        || mat2->rowcount != result -> rowcount){
        LogUtils::logErr("CrossProduct2D : Dimensional Mismatch");
        throw exception("CrossProduct2D : Dimensional Mismatch");
    }
    (void)crossProduct2D<<<gridSize, blockSize>>> (mat1, mat2, result);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callAddition(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                    MatrixOperations::Matrix2d *mat2) {
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, blockSize>>>(mat1, mat2);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callAddition(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                    MatrixOperations::Matrix2d *mat2, MatrixOperations::Matrix2d *result) {
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, blockSize>>>(mat1, mat2, result);
    (void)cudaDeviceSynchronize();
}


void MatrixOperations::callSigmoidActivation(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1) {
    (void)activationSigmoid<<<gridSize, blockSize>>>(mat1);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callSigmoidActivation(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                             MatrixOperations::Matrix2d *result) {
    (void)activationSigmoid<<<gridSize, blockSize>>>(mat1, result);
    (void)cudaDeviceSynchronize();
}


void MatrixOperations::callSigmoidDerivative(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1) {
    (void)derivativeSigmoid<<<gridSize,blockSize>>>(mat1);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callHardmardProduct(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                           MatrixOperations::Matrix2d *mat2) {
    (void) hadmardProduct<<<gridSize,blockSize>>>(mat1, mat2);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callConstantProduct2D(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1, float k) {
    (void) constantProduct2D<<<gridSize,blockSize>>>(mat1, k);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callTransit2D(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                     MatrixOperations::Matrix2d *result) {
    if (mat1->colcount != result->rowcount || mat1->rowcount != result->colcount){
        LogUtils::logErr("Transit2D : Dimensional Mismatch");
        throw exception("Transit2D : Dimensional Mismatch");
    }
    (void) transit2D<<<gridSize, blockSize>>>(mat1,result);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callSubtraction(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                       MatrixOperations::Matrix2d *mat2) {
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, blockSize>>>(mat1, mat2);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callSubtraction(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1,
                                    MatrixOperations::Matrix2d *mat2, MatrixOperations::Matrix2d *result) {
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, blockSize>>>(mat1, mat2, result);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callExponential(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1, float value) {
    (void) exponential2D<<<gridSize, blockSize>>>(mat1, value);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callExponential(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *mat1, float value,
                                       MatrixOperations::Matrix2d *result) {
    (void) exponential2D<<<gridSize, blockSize>>>(mat1, value,  result);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callMatCopy(dim3 gridSize, dim3 blockSize, MatrixOperations::Matrix2d *src,
                                   MatrixOperations::Matrix2d *dist) {
    (void) MatCpy<<<gridSize,  blockSize>>>(src,dist);
    (void) cudaDeviceSynchronize();
}




