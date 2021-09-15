//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_NETCONFIG_CUH
#define CUDANETWORKS_NETCONFIG_CUH


#include <string>

using namespace std;

static dim3 CUDA_BLOCK_SIZE = dim3(32, 32);
static float LEARNING_RATE = 0.05;

static const string DATA_PATH = R"(C:\Users\DanielSun\Desktop\resources\mnist\decompress_mnist)";
static const string TRAIN_PATH = "\\train";
static const string TEST_PATH = "\\test";

static int DATA_SIZE = 60000;
static int BATCH_SIZE = 100;
static int DATA_INPUT_SIZE_1D = 784;
static int DATA_INPUT_SIZE_2D_W = 28;
static int DATA_INPUT_SIZE_2D_H = 28;
static int DATA_OUTPUT_SIZE_1D = 10;

static int DATA_CATEGORIES = 10;
static int MNIST_INPUT_DIMENSION_ROW = 28;
static int MNIST_INPUT_DIMENSION_COL = 28;

static bool START_FROM_RANDOM = true;
class NetConfig {
public:
};



#endif //CUDANETWORKS_NETCONFIG_CUH
