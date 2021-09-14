//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_INPUTLAYERDENSE_CUH
#define CUDANETWORKS_INPUTLAYERDENSE_CUH

#include "../utils/MatrixOperations.cuh"
#include "../utils/LogUtils.cuh"
#include "Layer.cuh"

class InputLayerDense : public Layer{
public:
    InputLayerDense(int NODE_NUMBER){
        this->NODE_NUMBER = NODE_NUMBER;
        LogUtils::logInfo("input layer size : 1d : 784");
    }

    void calculateActivations(MatrixOperations::Matrix2d *prevNodes) override;

    void calculateErrorsAsOutput(MatrixOperations::Matrix2d *correctOut) override;

    void rememberErrors() override;

    void applyErrors() override;
};



#endif //CUDANETWORKS_INPUTLAYERDENSE_CUH
