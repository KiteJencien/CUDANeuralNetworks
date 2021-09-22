//
// Created by DanielSun on 9/14/2021.
//

#include "ModelMLP.cuh"
void ModelMLP::initializeRandomDense(vector<Layer*> layers, int beginIndex, int endIndex) {
    for(int i=beginIndex;i<endIndex;i++) {
        MatrixOperations::callAllocRandom(reinterpret_cast<DenseLayer*>(layers[i])->weights);
        MatrixOperations::callAllocRandom(reinterpret_cast<DenseLayer*>(layers[i])->bias);
        LogUtils::logInfo("Layer " + to_string(beginIndex + i) + " were random filled");
    }
}

void ModelMLP::trial(TrainData *input, int endIndex) {
    auto *inputLayer = reinterpret_cast<InputLayerDense*>(this->layers[0]);
    MatrixOperations::callFlatten(input->dataG,inputLayer->nodes);
    for (int i=1; i < endIndex; i++){
        auto *currentLayer = reinterpret_cast<DenseLayer*>(this->layers[i]);
        currentLayer->calculateActivations(this->layers[i-1]->nodes);
    }
}

void ModelMLP::reflect(TrainData *input, int endIndex) {
    auto *outputLayer = reinterpret_cast<DenseLayer*>(this->layers[endIndex-1]);
    outputLayer->calculateErrorsAsOutput(input->labelG);
    outputLayer->rememberErrors();
    for(int i=endIndex-2; i>0;i--){
        auto *currentLayer = reinterpret_cast<DenseLayer*>(this->layers[i]);
        DenseLayer::calculateErrors(currentLayer,this->layers[i+1]);
        currentLayer->rememberErrors();
    }
}

void ModelMLP::learn( int endIndex) {
    for(int i=1;i<endIndex;i++){
        auto *currentLayer = reinterpret_cast<DenseLayer*>(this->layers[i]);
        currentLayer->applyErrors();
        currentLayer->propagateWeights(layers[i-1]->nodes);
        currentLayer->propagateBias();
    }
}

float ModelMLP::calculateCost(TrainData *data, int endIndex, int* isCorrect) {
    MatrixOperations::Matrix2d* finalActivations;
    cudaMallocHost(reinterpret_cast<void**>(&finalActivations), sizeof(MatrixOperations::Matrix2d));
    auto *currentLayer = reinterpret_cast<DenseLayer*>(this->layers[endIndex-1]);
    cudaMallocHost(reinterpret_cast<void**>(&finalActivations->elements), DATA_OUTPUT_SIZE_1D*sizeof(float));
    finalActivations->rowcount = DATA_OUTPUT_SIZE_1D;
    finalActivations->colcount = 1;
    MatrixOperations::callMatCopyD2H(currentLayer->nodes,finalActivations);
    float out = 0;
    float max1 = 0;
    int maxIndex1 = 0;
    float max2 = 0;
    int maxIndex2 = 0;
    for(int  i=0; i<DATA_OUTPUT_SIZE_1D; i++){
        float act1 = *(finalActivations->elements + i);
        float act2 = *(data->label->elements + i);
        //cout<<act1<<" "<<act2<<endl;
        out += abs(act2-act1);
        if(max1 < act1){ max1 = act1; maxIndex1 = i;}
        if(max2 < act2){ max2 = act2; maxIndex2 = i;}
    }
    *isCorrect += maxIndex1 == maxIndex2 ? 1 : 0;
    cudaFree(finalActivations->elements);
    cudaFree(finalActivations);
    return out;
}

void ModelMLP::straightLearn(TrainData *input, int endIndex) {
    auto *outputLayer = reinterpret_cast<DenseLayer*>(this->layers[endIndex-1]);
    outputLayer->calculateErrorsAsOutput(input->labelG);
    outputLayer->rememberErrors();
    for(int i=endIndex-2; i>0;i--){
        auto *currentLayer = reinterpret_cast<DenseLayer*>(this->layers[i]);
        DenseLayer::calculateErrors(currentLayer,this->layers[i+1]);
    }
    //MatrixOperations::inspect(reinterpret_cast<DenseLayer*>(this->layers[endIndex-1])->errors);
    for(int i=1;i<endIndex;i++){
        auto *currentLayer = reinterpret_cast<DenseLayer*>(this->layers[i]);
        currentLayer->propagateWeights(layers[i-1]->nodes);
        currentLayer->propagateBias();
    }
    //MatrixOperations::inspect(reinterpret_cast<DenseLayer*>(this->layers[endIndex-1])->bias);
}

