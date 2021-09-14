//
// Created by DanielSun on 9/14/2021.
//
#include "../NetConfig.cuh"
#include "MatrixOperations.cuh"
using namespace std;

//global functions: operations

__global__ void setElement(MatrixOperations::Matrix2d *mat1, int row, int col, float value){
    MatrixOperations::setElement2D(mat1,row, col, value);
}

__global__ void allocateElements(MatrixOperations::Matrix2d *mat1,int row, int col){
    mat1->rowcount = row;
    mat1->colcount = col;
    (void)cudaMalloc(reinterpret_cast<void**>(&mat1->elements),5*sizeof(float));
}
//random number fill (0-1)
__global__ void allocRandom(long seed, MatrixOperations::Matrix2d *mat1){
    curandStateXORWOW_t state;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(row*col*seed,0,0,&state);
    MatrixOperations::setElement2D(mat1, row, col, static_cast<float>((curand_uniform(&state)*2.0F) -1.0F));
}

//zero fill
__global__ void allocZero( MatrixOperations::Matrix2d *mat1){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    MatrixOperations::setElement2D(mat1, row, col, 0.0F);
}

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

void MatrixOperations::callCrossProduct2D( MatrixOperations::Matrix2d *mat1,
                                           MatrixOperations::Matrix2d *mat2, MatrixOperations::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    //condition
    if (mat1->colcount != mat2->rowcount || mat1->colcount != result -> colcount
        || mat2->rowcount != result -> rowcount){
        LogUtils::logErr("CrossProduct2D : Dimensional Mismatch");
        throw exception("CrossProduct2D : Dimensional Mismatch");
    }


    (void)crossProduct2D<<<gridSize, CUDA_BLOCK_SIZE>>> (mat1, mat2, result);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callAddition( MatrixOperations::Matrix2d *mat1,
                                     MatrixOperations::Matrix2d *mat2) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callAddition(MatrixOperations::Matrix2d *mat1,
                                    MatrixOperations::Matrix2d *mat2, MatrixOperations::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2, result);
    (void)cudaDeviceSynchronize();
}


void MatrixOperations::callSigmoidActivation( MatrixOperations::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void)activationSigmoid<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callSigmoidActivation( MatrixOperations::Matrix2d *mat1,
                                              MatrixOperations::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void)activationSigmoid<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result);
    (void)cudaDeviceSynchronize();
}


void MatrixOperations::callSigmoidDerivative(MatrixOperations::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void)derivativeSigmoid<<<gridSize,CUDA_BLOCK_SIZE>>>(mat1);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callHardmardProduct(MatrixOperations::Matrix2d *mat1,
                                           MatrixOperations::Matrix2d *mat2) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) hadmardProduct<<<gridSize,CUDA_BLOCK_SIZE>>>(mat1, mat2);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callConstantProduct2D( MatrixOperations::Matrix2d *mat1, float k) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) constantProduct2D<<<gridSize,CUDA_BLOCK_SIZE>>>(mat1, k);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callTransit2D( MatrixOperations::Matrix2d *mat1,
                                      MatrixOperations::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    if (mat1->colcount != result->rowcount || mat1->rowcount != result->colcount){
        LogUtils::logErr("Transit2D : Dimensional Mismatch");
        throw exception("Transit2D : Dimensional Mismatch");
    }
    (void) transit2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1,result);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callSubtraction(MatrixOperations::Matrix2d *mat1,
                                       MatrixOperations::Matrix2d *mat2) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callSubtraction( MatrixOperations::Matrix2d *mat1,
                                        MatrixOperations::Matrix2d *mat2, MatrixOperations::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    if (mat1 -> colcount != mat2 -> colcount || mat1 -> rowcount != mat2 -> rowcount){
        LogUtils::logErr("Addition2D : Dimensional Mismatch");
        throw exception("Addition2D : Dimensional Mismatch");
    }
    (void)addition2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2, result);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callExponential(MatrixOperations::Matrix2d *mat1, float value) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) exponential2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, value);
    (void)cudaDeviceSynchronize();
}

void MatrixOperations::callExponential( MatrixOperations::Matrix2d *mat1, float value,
                                        MatrixOperations::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) exponential2D<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, value,  result);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callMatCopy( MatrixOperations::Matrix2d *src,
                                    MatrixOperations::Matrix2d *dist) {
    dim3 gridSize = dim3((src->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (src->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) MatCpy<<<gridSize,  CUDA_BLOCK_SIZE>>>(src,dist);
    (void) cudaDeviceSynchronize();
}



void MatrixOperations::callAllocElement(MatrixOperations::Matrix2d *mat1, int row, int col) {
    (void) allocateElements<<<dim3(1,1), CUDA_BLOCK_SIZE>>>(mat1, row, col);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callAllocZero(MatrixOperations::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) allocZero<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callAllocRandom(MatrixOperations::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) allocRandom<<<gridSize, CUDA_BLOCK_SIZE>>>(time(nullptr),mat1);
    (void) cudaDeviceSynchronize();
}

void MatrixOperations::callSetElement(MatrixOperations::Matrix2d *mat1, int row, int col, float value) {
    (void) setElement<<<dim3(1,1),CUDA_BLOCK_SIZE>>>(mat1, row, col, value);
    (void)cudaDeviceSynchronize();
}