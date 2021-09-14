//
// Created by DanielSun on 9/14/2021.
//

#include "ModelMLP.cuh"
void ModelMLP::initializeRandomDense(vector<Layer*> layers, int beginIndex, int endIndex) {
    for(int i=beginIndex;i<endIndex;i++) {
        MatrixOperations::callAllocRandom(dynamic_cast<DenseLayer*>(layers[i])->weights);
        MatrixOperations::callAllocRandom(dynamic_cast<DenseLayer*>(layers[i])->bias);
        LogUtils::logInfo("Layer " + to_string(beginIndex + i) + " were random filled");
    }
}
