//
// Created by DanielSun on 9/14/2021.
//

#include "DenseLayer.cuh"
#include "../NetConfig.cuh"

void DenseLayer::calculateActivations(MatrixOperations::Matrix2d *prevNodes)  {
    //z = w * a + b , a1 = sigmoid(z)
    MatrixOperations::callCrossProduct2D( this->weights, prevNodes,this->z);
    MatrixOperations::callAddition(this->z,this->bias);
    MatrixOperations::callSigmoidActivation(this->z, this->nodes);
}

// Error = (aL - y) *hadmard* sigmoidDerivative(z)
void DenseLayer::calculateErrorsAsOutput(MatrixOperations::Matrix2d *correctOut)  {
    MatrixOperations::callSigmoidDerivative( this->z);
    MatrixOperations::callSubtraction(this->z, correctOut, this->errors);
    //result will be stored in "errors"
    MatrixOperations::callHardmardProduct( this->errors, this->z);
}

//deltaWeight = ErrorL * (a(L-1))^T
//apply propagate : Weights - deltaWeights
void DenseLayer::propagateWeights(MatrixOperations::Matrix2d *prevActivations)  {
    MatrixOperations::Matrix2d *transit;
    (void) cudaMallocManaged(reinterpret_cast<void**>(&transit), sizeof(MatrixOperations::Matrix2d));
    MatrixOperations::callAllocElementD(transit, 1, prevActivations->rowcount);
    MatrixOperations::callTransit2D( prevActivations, transit);
    MatrixOperations::callCrossProduct2D( this->errors, transit, this->weightDerivative);
    MatrixOperations::callConstantProduct2D( this->weightDerivative, LEARNING_RATE);
    MatrixOperations::callSubtraction( this->weights, this->weightDerivative);
    (void) cudaFree(transit->elements);
    (void) cudaFree(transit);
}

// E^l = (W^(l+1))^T * E^(l+1) <Hardmard> sigmoid'(z^l)
void DenseLayer::calculateErrors(DenseLayer *thisLayer, MatrixOperations::Matrix2d *nextLayerWeights, MatrixOperations::Matrix2d *nextLayerErr)  {
    MatrixOperations::Matrix2d *transitWeights;
    (void) cudaMallocManaged(reinterpret_cast<void**>(&transitWeights), sizeof(MatrixOperations::Matrix2d));
    MatrixOperations::callAllocElementD(transitWeights, nextLayerWeights->colcount, nextLayerWeights->rowcount);
    MatrixOperations::callTransit2D( nextLayerWeights, transitWeights);
    MatrixOperations::callCrossProduct2D( transitWeights, nextLayerErr, thisLayer->errors);
    MatrixOperations::callSigmoidDerivative( thisLayer->z);
    MatrixOperations::callHardmardProduct( thisLayer->errors, thisLayer->z);
    (void) cudaFree(transitWeights->elements);
    (void) cudaFree(transitWeights);
}

void DenseLayer::rememberErrors() {
    MatrixOperations::callAddition( this->pastErrors, this->errors);
}

void DenseLayer::propagateBias() {
    MatrixOperations::callConstantProduct2D( this->errors, LEARNING_RATE);
    MatrixOperations::callSubtraction( this->bias, this->errors);
    MatrixOperations::callConstantProduct2D( this->errors, 1.0F/LEARNING_RATE);
}

void DenseLayer::applyErrors() {
    MatrixOperations::callConstantProduct2D( this->pastErrors, 1.0F/static_cast<float>(BATCH_SIZE));
    MatrixOperations::callMatCopy( this->pastErrors, this->errors);
    MatrixOperations::callAllocZero(this->pastErrors);
}

void DenseLayer::calculateErrors(Layer *in, Layer *next) {
    //do not call this method for layers that are not dense layers
    auto* thisLayer = dynamic_cast<DenseLayer*>(in);
    auto* nextLayer = dynamic_cast<DenseLayer*>(next);
    calculateErrors(thisLayer, nextLayer->weights, nextLayer->errors);
}







