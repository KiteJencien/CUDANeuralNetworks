//
// Created by DanielSun on 9/14/2021.
//


#include "TrainData.cuh"

void TrainData::AllocateDataOnDeviceMem() {

    (void) cudaMallocManaged(reinterpret_cast<void**>(&dataG), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMallocManaged(reinterpret_cast<void**>(&labelG), sizeof(MatrixOperations::Matrix2d));

    MatrixOperations::callAllocElementD(dataG, DATA_INPUT_SIZE_1D, 1);
    MatrixOperations::callAllocElementD(labelG, DATA_OUTPUT_SIZE_1D, 1);

    MatrixOperations::callMatCopy(data,dataG);
    MatrixOperations::callMatCopy(label,labelG);
}

void TrainData::freeDataOnDeviceMem() const {
    (void) cudaFree(dataG->elements);
    (void) cudaFree(labelG->elements);
    (void) cudaFree(dataG);
    (void) cudaFree(labelG);
}
