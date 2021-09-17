//
// Created by DanielSun on 9/14/2021.
//

#include "Dataset.cuh"
#define random(x) rand()%(x)

static bool contains(int* set, int size, int element){
    for (int j = 0; j < size; j++){
        if(element == set[j]){
            return true;
        }
    }
    return false;
}

void Dataset::readDatasetMNIST(Dataset::DatasetInstance *instance, const string& type) {
    LogUtils::logInfo("----------< Reading Dataset : MNIST >---------",0x05);
    int data_id = 0;
     for(int i=0; i < DATA_CATEGORIES; i++){
         for(int j=0; j<DATA_SIZE; j++){
             string path = DATA_PATH + TRAIN_PATH;
             path.append("\\").append(to_string(i)).append("\\").append(to_string(j)).append(".bmp");
             auto* content = new TrainData();
             if(!BMPDecomp::forceReadBMP(content->data, path.c_str())) break;
             MatrixOperations::callAllocZero(content->label);
             MatrixOperations::callSetElement(content->label, i, 0, 1.0F);
             instance->data[data_id] = content;
             data_id++;
         }
         LogUtils::logInfo("Data read complete :" + to_string(data_id) + " for label " + to_string(i),0x06);
     }
}

Dataset::BatchInstance *Dataset::generateBatch(Dataset::DatasetInstance *instance) {
    srand((int)time(0));
    auto* batch = new BatchInstance();
    int* batchIndexes;
    cudaMallocHost(reinterpret_cast<void**>(&batchIndexes),BATCH_SIZE*sizeof(int));
    for (int j = 0; j < BATCH_SIZE; j++){
        batchIndexes[j] = -1;
    }
    for (int i=0; i<BATCH_SIZE; i++){
        int index = random(DATA_SIZE);
        while(contains(batchIndexes,BATCH_SIZE,index)){
            index = random(DATA_SIZE);
        }
        TrainData* data = instance->data[index];
        data->AllocateDataOnDeviceMem();
        batch->data[i] = data;
        batchIndexes[i] = index;
    }
    cudaFreeHost(batchIndexes);
    return batch;
}

void Dataset::freeBatch(Dataset::BatchInstance *instance) {
    for(TrainData* d : instance->data){
        //EXCEPT THE CASE IF REPEATED DATASETS EXIST
        d->freeDataOnDeviceMem();
        cudaFree(d);
    }
    cudaFree(instance);
}
