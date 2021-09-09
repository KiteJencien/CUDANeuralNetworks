//
// Created by DanielSun on 9/7/2021.
//

#ifndef NETWORKS_TRAINDATA_CUH
#define NETWORKS_TRAINDATA_CUH

#include "../NetConfig.cuh"
#include "../utils/MatrixOperations.cuh"
#include "../utils/LogUtils.cuh"

class TrainData {
public:
     MatrixOperations::Matrix2d *data;
     MatrixOperations::Matrix2d *label;

     //these will not refer to anything if the data is not in a running batch
     MatrixOperations::Matrix2d *dataG;
     MatrixOperations::Matrix2d *labelG;

     //place data on the host while initiating
     TrainData(){
         (void) cudaMallocHost(reinterpret_cast<void**>(&data), sizeof(MatrixOperations::Matrix2d));
         (void) cudaMallocHost(reinterpret_cast<void**>(&data->elements), DATA_INPUT_SIZE_1D * sizeof(float));
         (void) cudaMallocHost(reinterpret_cast<void**>(&label), sizeof(MatrixOperations::Matrix2d));
         (void) cudaMallocHost(reinterpret_cast<void**>(&label->elements), DATA_OUTPUT_SIZE_1D * sizeof(float));
     }

     //copy the dataset to the device memory
     void AllocateDataOnDeviceMem();

     //clear the dataset on the device memory
     void FreeDataOnDeviceMem() const;


};


#endif //NETWORKS_TRAINDATA_CUH
