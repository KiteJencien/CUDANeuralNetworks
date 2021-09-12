//
// Created by DanielSun on 9/7/2021.
//

#include "TrainData.cuh"

void TrainData::AllocateDataOnDeviceMem() {

    (void) cudaMallocManaged(reinterpret_cast<void**>(&dataG), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMallocManaged(reinterpret_cast<void**>(&labelG),  sizeof(MatrixOperations::Matrix2d));

    MatrixOperations::callAllocElement(dataG, DATA_INPUT_SIZE_1D, 1);
    MatrixOperations::callAllocElement(labelG, DATA_INPUT_SIZE_1D, 1);

    MatrixOperations::callMatCopy(data,dataG);
    MatrixOperations::callMatCopy(label,labelG);
}

void TrainData::FreeDataOnDeviceMem() const {
    (void) cudaFree(dataG->elements);
    (void) cudaFree(labelG->elements);
}
