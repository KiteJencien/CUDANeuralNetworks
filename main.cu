#include <iostream>
#include "utils/MatrixOperations.cuh"
#include "Models/ModelMLP.cuh"
#include "NetConfig.cuh"
#include "dataset/Dataset.cuh"
#include <cstdio>

using namespace std;

int main(int argc, char** argv) {
    new ModelMLP;
    Dataset::DatasetInstance *instance = new Dataset::DatasetInstance;
    Dataset::readDatasetMNIST(instance,TRAIN_PATH);
    return 0;
}

