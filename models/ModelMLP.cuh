//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_MODELMLP_CUH
#define CUDANETWORKS_MODELMLP_CUH


#include "../layers/Layer.cuh"
#include "../layers/InputLayerDense.cuh"
#include "../layers/DenseLayer.cuh"
#include "../NetConfig.cuh"
#include "vector"
#include "../datasets/TrainData.cuh"

class ModelMLP {
public:
    vector<Layer*> layers = vector<Layer*>(4);
    ModelMLP(){
        LogUtils::logInfo("----------< Registering Model : MLP >---------",0x05);
        layers[0] = new InputLayerDense(784);
        layers[1] = new DenseLayer(16, 784);
        layers[2] = new DenseLayer(16, 16);
        layers[3] = new DenseLayer(10, 16);
        if (START_FROM_RANDOM) initializeRandomDense(layers, 1, 4);
    }

    static void initializeRandomDense(vector<Layer*> layers, int beginIndex,int endIndex);

    void trial(TrainData* input, int endIndex);

    void reflect(TrainData *input, int endIndex);

    void learn( int endIndex);

    float calculateCost(TrainData *data, int endIndex, int* isCorrect);
};



#endif //CUDANETWORKS_MODELMLP_CUH
