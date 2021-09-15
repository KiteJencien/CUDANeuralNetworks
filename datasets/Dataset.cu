//
// Created by DanielSun on 9/14/2021.
//

#include "Dataset.cuh"

void Dataset::readDatasetMNIST(Dataset::DatasetInstance *instance, const string& type) {
    LogUtils::logInfo("----------< Reading Dataset : MNIST >---------",0x05);
    int data_id = 0;
     for(int i=0; i < DATA_CATEGORIES; i++){
         for(int j=0; j<DATA_SIZE; j++){
             string path = DATA_PATH + TRAIN_PATH;
             path.append("\\").append(to_string(i)).append("\\").append(to_string(j)).append(".bmp");
             auto* output = new TrainData();
             if(!BMPDecomp::ReadBmp(output->data,path.c_str())) break;
             MatrixOperations::callAllocZero( output->label);
             MatrixOperations::callSetElement( output->label, i,0,1.0F);
             instance->data[data_id] = output;
             data_id++;
         }
         LogUtils::logInfo("Data read complete :" + to_string(data_id) + " for label " + to_string(i),0x01);
     }
}