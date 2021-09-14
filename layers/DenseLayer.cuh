//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_DENSELAYER_CUH
#define CUDANETWORKS_DENSELAYER_CUH

#include "../utils/MatrixOperations.cuh"
#include "../utils/LogUtils.cuh"
#include "Layer.cuh"

class DenseLayer : public Layer{
public:
    MatrixOperations::Matrix2d *weights, *weightDerivative,*bias, *z, *nodes, *errors, *pastErrors;

    //the callback function for error calculation
    typedef void (callBack)(Layer *in, Layer *next);
    static void calculateErrors(Layer *thisLayer, Layer *nextLayer);

    //construct the layer object
    DenseLayer(int NODE_NUMBER, int PREV_NODE_NUMBER){
        this->NODE_NUMBER = NODE_NUMBER;

        //Allocate structs
        (void) cudaMallocManaged(reinterpret_cast<void**>(&weights),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMallocManaged(reinterpret_cast<void**>(&weightDerivative),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMallocManaged(reinterpret_cast<void**>(&bias),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMallocManaged(reinterpret_cast<void**>(&z),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMallocManaged(reinterpret_cast<void**>(&nodes),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMallocManaged(reinterpret_cast<void**>(&errors),sizeof(MatrixOperations::Matrix2d));
        (void) cudaMallocManaged(reinterpret_cast<void**>(&pastErrors),sizeof(MatrixOperations::Matrix2d));

        //Allocate elements
        MatrixOperations::callAllocElement(weights, PREV_NODE_NUMBER, NODE_NUMBER);
        MatrixOperations::callAllocElement(weightDerivative, PREV_NODE_NUMBER, NODE_NUMBER);
        MatrixOperations::callAllocElement(bias, NODE_NUMBER, 1);
        MatrixOperations::callAllocElement(nodes, NODE_NUMBER, 1);
        MatrixOperations::callAllocElement(z, NODE_NUMBER, 1);
        MatrixOperations::callAllocElement(errors, NODE_NUMBER, 1);
        MatrixOperations::callAllocElement(pastErrors, NODE_NUMBER, 1);

        LogUtils::logInfo("layer registered : dense : " + to_string(NODE_NUMBER) +
                          " : layer memory usage : " + to_string(static_cast<float>(NODE_NUMBER*(PREV_NODE_NUMBER)*sizeof(float))
                                                                 / (1024.0F*1024.0F)) + " MB");
    }

    //The front operation of running the network
    void calculateActivations(MatrixOperations::Matrix2d *prevNodes) override ;

    //this is used for straight training
    static void calculateErrors(DenseLayer *thisLayer, MatrixOperations::Matrix2d *nextLayerWeights, MatrixOperations::Matrix2d *nextLayerErr) ;

    //this is used for prepare batch training
    void rememberErrors();

    //this is for apply batch training
    void applyErrors();

    //same as above but for the layer been used as outputs
    void calculateErrorsAsOutput(MatrixOperations::Matrix2d *correctOut) ;

    void propagateWeights(MatrixOperations::Matrix2d *prevActivations) ;

    void propagateBias();
};


#endif //CUDANETWORKS_DENSELAYER_CUH
