#include <iostream>
#include "utils/MatrixOperations.cuh"
#include "models/ModelMLP.cuh"
#include "NetConfig.cuh"
#include "datasets/Dataset.cuh"
#include <cstdio>

using namespace std;

int main(int argc, char** argv) {
    new ModelMLP;
    auto *instance = new Dataset::DatasetInstance;
    Dataset::readDatasetMNIST(instance,TRAIN_PATH);
    return 0;
}

