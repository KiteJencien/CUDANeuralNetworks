//
// Created by DanielSun on 8/31/2021.
//

#include "DenseLayer.cuh"
#include "../NetConfig.cuh"

void DenseLayer::calculateActivations(MatrixOperations::Matrix2d *prevNodes)  {
    dim3 gridSize = dim3((this->weights->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
    (this->weights->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    //z = w * a + b , a1 = sigmoid(z)
    MatrixOperations::callCrossProduct2D(gridSize, CUDA_BLOCK_SIZE, this->weights, prevNodes,this->z);
    MatrixOperations::callAddition(gridSize,CUDA_BLOCK_SIZE,this->z,this->bias);
    MatrixOperations::callSigmoidActivation(gridSize, CUDA_BLOCK_SIZE,this->z, this->nodes);
}

// Error = (aL - y) *hadmard* sigmoidDerivative(z)
void DenseLayer::calculateErrorsAsOutput(MatrixOperations::Matrix2d *correctOut)  {
    dim3 gridSize = dim3((this->nodes->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                    (this->nodes->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::callSigmoidDerivative(gridSize, CUDA_BLOCK_SIZE, this->z);
    MatrixOperations::callSubtraction(gridSize,CUDA_BLOCK_SIZE,this->z, correctOut, this->errors);
    //result will be stored in "errors"
    MatrixOperations::callHardmardProduct(gridSize, CUDA_BLOCK_SIZE, this->errors, this->z);
}

//deltaWeight = ErrorL * (a(L-1))^T
//apply propagate : Weights - deltaWeights
void DenseLayer::propagateWeights(MatrixOperations::Matrix2d *prevActivations)  {
    dim3 gridSize = dim3((this->weights->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (this->weights->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::Matrix2d *transit;
    (void) cudaMalloc(reinterpret_cast<void**>(&transit), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMalloc(reinterpret_cast<void**>(&transit->elements), prevActivations->rowcount * sizeof(float));
    transit->colcount = prevActivations->rowcount;
    transit->rowcount = 1;
    MatrixOperations::callTransit2D(gridSize, CUDA_BLOCK_SIZE, prevActivations, transit);
    MatrixOperations::callCrossProduct2D(gridSize, CUDA_BLOCK_SIZE, this->errors, transit, this->weightDerivative);
    MatrixOperations::callConstantProduct2D(gridSize, CUDA_BLOCK_SIZE, this->weightDerivative, LEARNING_RATE);
    MatrixOperations::callSubtraction(gridSize, CUDA_BLOCK_SIZE, this->weights, this->weightDerivative);
    (void) cudaFree(transit->elements);
    (void) cudaFree(transit);
}

void DenseLayer::calculateErrors(MatrixOperations::Matrix2d *nextLayerWeights, MatrixOperations::Matrix2d *nextLayerErr)  {
    dim3 gridSize = dim3((nextLayerWeights->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (nextLayerWeights->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::Matrix2d *transitWeights;
    (void) cudaMalloc(reinterpret_cast<void**>(&transitWeights), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMalloc(reinterpret_cast<void**>(&transitWeights->elements),
                             nextLayerWeights->rowcount * nextLayerWeights->colcount * sizeof(float));
    transitWeights->rowcount = nextLayerWeights->colcount;
    transitWeights->colcount = nextLayerWeights->rowcount;
    MatrixOperations::callTransit2D(gridSize, CUDA_BLOCK_SIZE, nextLayerWeights, transitWeights);
    MatrixOperations::callCrossProduct2D(gridSize, CUDA_BLOCK_SIZE, transitWeights, nextLayerWeights, this->errors);
    MatrixOperations::callSigmoidDerivative(gridSize, CUDA_BLOCK_SIZE, this->z);
    MatrixOperations::callHardmardProduct(gridSize, CUDA_BLOCK_SIZE, this->errors, this->z);
    (void) cudaFree(transitWeights->elements);
    (void) cudaFree(transitWeights);
}

void DenseLayer::rememberErrors() {
    dim3 gridSize = dim3((this->errors->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (this->errors->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::callAddition(gridSize, CUDA_BLOCK_SIZE, this->pastErrors, this->errors);
}

void DenseLayer::propagateBias() {
    dim3 gridSize = dim3((this->nodes->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (this->nodes->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::callConstantProduct2D(gridSize, CUDA_BLOCK_SIZE, this->errors, LEARNING_RATE);
    MatrixOperations::callSubtraction(gridSize, CUDA_BLOCK_SIZE, this->bias, this->errors);
    MatrixOperations::callConstantProduct2D(gridSize, CUDA_BLOCK_SIZE, this->errors, 1.0F/LEARNING_RATE);
}

void DenseLayer::applyErrors() {
    dim3 gridSize = dim3((this->errors->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (this->errors->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::callConstantProduct2D(gridSize, CUDA_BLOCK_SIZE, this->pastErrors, 1.0F/LEARNING_RATE);
    MatrixOperations::callMatCopy(gridSize, CUDA_BLOCK_SIZE, this->pastErrors, this->errors);
}

void DenseLayer::calculateErrors(DenseLayer *thisLayer, DenseLayer *nextLayer) {
    MatrixOperations::Matrix2d *nextLayerWeights = nextLayer->weights;
    dim3 gridSize = dim3((nextLayerWeights->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (nextLayerWeights->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    MatrixOperations::Matrix2d *transitWeights;
    (void) cudaMalloc(reinterpret_cast<void**>(&transitWeights), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMalloc(reinterpret_cast<void**>(&transitWeights->elements),
                             nextLayerWeights->rowcount * nextLayerWeights->colcount * sizeof(float));
    transitWeights->rowcount = nextLayerWeights->colcount;
    transitWeights->colcount = nextLayerWeights->rowcount;
    MatrixOperations::callTransit2D(gridSize, CUDA_BLOCK_SIZE, nextLayerWeights, transitWeights);
    MatrixOperations::callCrossProduct2D(gridSize, CUDA_BLOCK_SIZE, transitWeights, nextLayerWeights, thisLayer->errors);
    MatrixOperations::callSigmoidDerivative(gridSize, CUDA_BLOCK_SIZE, thisLayer->z);
    MatrixOperations::callHardmardProduct(gridSize, CUDA_BLOCK_SIZE, thisLayer->errors, thisLayer->z);
    (void) cudaFree(transitWeights->elements);
    (void) cudaFree(transitWeights);
}







