//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_DATASET_CUH
#define CUDANETWORKS_DATASET_CUH

#include "TrainData.cuh"
#include "../utils/BMPDecomp.cuh"
#include "../NetConfig.cuh"
#include <string>



class Dataset {
public:
    struct DatasetInstance{
        vector<TrainData*> data = vector<TrainData*>(DATA_SIZE);
    };

    struct BatchInstance{
        vector<TrainData*> data = vector<TrainData*>(BATCH_SIZE);
    };

    //read dataset
    static void readDatasetMNIST(DatasetInstance *instance, const string& type);

    //copy a batch of data to cuda memory
    static BatchInstance* generateBatch(DatasetInstance* instance);

    static void freeBatch(BatchInstance* instance);

};



#endif //CUDANETWORKS_DATASET_CUH
