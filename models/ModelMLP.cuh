//
// Created by DanielSun on 9/3/2021.
//

#ifndef NETWORKS_MODELMLP_CUH
#define NETWORKS_MODELMLP_CUH


#include "../layers/Layer.cuh"
#include "../layers/InputLayerDense.cuh"
#include "../layers/DenseLayer.cuh"

class ModelMLP {
public:
    Layer* layers[4];
    ModelMLP(){
        LogUtils::logInfo("----------< Registering Model : MLP >---------",0x05);
        layers[0] = new InputLayerDense(784);
        layers[1] = new DenseLayer(16, 784);
        layers[2] = new DenseLayer(16, 16);
        layers[3] = new DenseLayer(10, 16);
    }
};


#endif //NETWORKS_MODELMLP_CUH
