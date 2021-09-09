//
// Created by DanielSun on 8/31/2021.
//

#ifndef NETWORKS_DENSELAYER_CUH
#define NETWORKS_DENSELAYER_CUH

#include "../utils/MatrixOperations.cuh"
#include "../utils/LogUtils.cuh"
#include "Layer.cuh"

class DenseLayer : public Layer{
public:
    MatrixOperations::Matrix2d *weights, *weightDerivative,*bias, *z, *nodes, *errors, *pastErrors;

    //the callback function for error calculation
    typedef void (callBack)(DenseLayer *thisLayer, DenseLayer *nextLayer);
    static void calculateErrors(DenseLayer *thisLayer, DenseLayer *nextLayer);

    //construct the layer object
    DenseLayer(int NODE_NUMBER, int PREV_NODE_NUMBER){
        this->NODE_NUMBER = NODE_NUMBER;
        int sizeWeights = NODE_NUMBER * PREV_NODE_NUMBER * static_cast<int>(sizeof(float));
        int sizeNormal = NODE_NUMBER * static_cast<int>(sizeof(float));

        //Allocate structs
        (void) cudaMalloc(reinterpret_cast<void**>(&weights),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMalloc(reinterpret_cast<void**>(&weightDerivative),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMalloc(reinterpret_cast<void**>(&bias),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMalloc(reinterpret_cast<void**>(&z),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMalloc(reinterpret_cast<void**>(&nodes),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMalloc(reinterpret_cast<void**>(&errors),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMalloc(reinterpret_cast<void**>(&pastErrors),sizeof(MatrixOperations::Matrix2d));

        //Allocate elements
        (void) cudaMalloc(reinterpret_cast<void**>(&weights->elements),sizeWeights);
        (void) cudaMalloc(reinterpret_cast<void**>(&weightDerivative->elements),sizeWeights);
        (void) cudaMalloc(reinterpret_cast<void**>(&bias->elements),sizeNormal);
        (void) cudaMalloc(reinterpret_cast<void**>(&z->elements),sizeNormal);
        (void) cudaMalloc(reinterpret_cast<void**>(&nodes->elements),sizeNormal);
        (void) cudaMalloc(reinterpret_cast<void**>(&errors->elements),sizeNormal);
        (void) cudaMalloc(reinterpret_cast<void**>(&pastErrors->elements),sizeNormal);

        weights->rowcount = PREV_NODE_NUMBER;
        weights->colcount = NODE_NUMBER;
        weightDerivative->rowcount = PREV_NODE_NUMBER;
        weightDerivative->colcount = NODE_NUMBER;
        bias->rowcount = NODE_NUMBER;
        bias->colcount = 1;
        nodes->rowcount = NODE_NUMBER;
        nodes->colcount = 1;
        z->rowcount = NODE_NUMBER;
        z->colcount = 1;
        errors->rowcount = NODE_NUMBER;
        errors->colcount = 1;
        pastErrors->rowcount = NODE_NUMBER;
        pastErrors->colcount = 1;

        LogUtils::logInfo("layer registered : dense : " + to_string(NODE_NUMBER) +
              " : layer memory usage : " + to_string(static_cast<float>(NODE_NUMBER*(PREV_NODE_NUMBER)*sizeof(float))
              / (1024.0F*1024.0F)) + " MB");
    }

    //The front operation of running the network
    void calculateActivations(MatrixOperations::Matrix2d *prevNodes) override ;

    void calculateErrors(Layer *nextLayer);

    //this is used for straight training
    void calculateErrors(MatrixOperations::Matrix2d *nextLayerWeights, MatrixOperations::Matrix2d *nextLayerErr) ;

    //this is used for prepare batch training
    void rememberErrors();

    //this is for apply batch training
    void applyErrors();

    //same as above but for the layer been used as outputs
    void calculateErrorsAsOutput(MatrixOperations::Matrix2d *correctOut) ;

    void propagateWeights(MatrixOperations::Matrix2d *prevActivations) ;

    void propagateBias();
};


#endif //NETWORKS_DENSELAYER_CUH
