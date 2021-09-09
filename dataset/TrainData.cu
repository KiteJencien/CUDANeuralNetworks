//
// Created by DanielSun on 9/7/2021.
//

#include "TrainData.cuh"

void TrainData::AllocateDataOnDeviceMem() {

    (void) cudaMalloc(reinterpret_cast<void**>(&dataG), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMalloc(reinterpret_cast<void**>(&dataG->elements), DATA_INPUT_SIZE_1D * sizeof(float));
    (void) cudaMalloc(reinterpret_cast<void**>(&labelG), sizeof(MatrixOperations::Matrix2d));
    (void) cudaMalloc(reinterpret_cast<void**>(&labelG->elements), DATA_OUTPUT_SIZE_1D * sizeof(float));

    // dist, src, size, mode
    (void) cudaMemcpy(dataG,data, sizeof(MatrixOperations::Matrix2d), cudaMemcpyHostToDevice);
    (void) cudaMemcpy(dataG->elements, data->elements, data->colcount * data->rowcount * sizeof(float), cudaMemcpyHostToDevice);
    (void) cudaMemcpy(labelG, label, sizeof(MatrixOperations::Matrix2d), cudaMemcpyHostToDevice);
    (void) cudaMemcpy(labelG->elements, label->elements, label->rowcount * label -> colcount * sizeof(float), cudaMemcpyHostToDevice);
}

void TrainData::FreeDataOnDeviceMem() const {
    (void) cudaFree(dataG->elements);
    (void) cudaFree(dataG);
}
