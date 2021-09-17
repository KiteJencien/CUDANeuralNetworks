#include <iostream>
#include "utils/MatrixOperations.cuh"
#include "models/ModelMLP.cuh"
#include "NetConfig.cuh"
#include "datasets/Dataset.cuh"
#include <cstdio>

using namespace std;

static void runModelMLP(){
    auto *model = new ModelMLP;
    auto *instance = new Dataset::DatasetInstance;
    Dataset::readDatasetMNIST(instance,TRAIN_PATH);
    for(int i=0; i<1000; i++){
        float cost=0;
        int isCorrect=0;
        Dataset::BatchInstance* batch = Dataset::generateBatch(instance);
        for(int j=0;j<BATCH_SIZE;j++){
            model->trial(batch->data[j],4);
            cost += model->calculateCost(batch->data[j],4, &isCorrect);
            model->reflect(batch->data[j],4);
        }
        model->learn(4);
        // MatrixOperations::inspect(model->layers[3]->nodes);
        LogUtils::logInfo("Training : current cost : " + to_string(cost/static_cast<float>(BATCH_SIZE)
        ) + " at batch " + to_string(i) + " ACCURACY :" + to_string((float)isCorrect/(float)BATCH_SIZE), 0X05);
        Dataset::freeBatch(batch);
    }
}

int main(int argc, char** argv) {
    runModelMLP();
    return 0;
}

