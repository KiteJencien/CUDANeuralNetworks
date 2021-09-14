//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_LAYER_CUH
#define CUDANETWORKS_LAYER_CUH

#include "../utils/MatrixOperations.cuh"

class Layer {

public:
    MatrixOperations::Matrix2d *nodes;
    int NODE_NUMBER;

    static void calculateErrors(void (*callback)(Layer*,Layer*), Layer *thisLayer, Layer *nextLayer){
        (*callback)(thisLayer, nextLayer);
    };

    //forward calculate
    virtual void calculateActivations(MatrixOperations::Matrix2d *prevNodes) = 0;

    virtual void calculateErrorsAsOutput(MatrixOperations::Matrix2d *correctOut) = 0;

    virtual void rememberErrors() = 0;

    virtual void applyErrors() = 0;
};



#endif //CUDANETWORKS_LAYER_CUH
