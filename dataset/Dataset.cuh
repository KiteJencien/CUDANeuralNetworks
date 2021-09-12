//
// Created by DanielSun on 9/9/2021.
//

#ifndef NETWORKS_DATASET_CUH
#define NETWORKS_DATASET_CUH


#include "TrainData.cuh"
#include "../NetConfig.cuh"

class Dataset {
public:
    struct DatasetInstance{
        int size = DATA_SIZE;
        TrainData* data;
    };

    struct BatchInstance{
        int size = BATCH_SIZE;
        TrainData* data;
    };

    Dataset(){

    }

};


#endif //NETWORKS_DATASET_CUH
