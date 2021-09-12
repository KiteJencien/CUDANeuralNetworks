//
// Created by DanielSun on 8/31/2021.
//

#ifndef NETWORKS_NETCONFIG_CUH
#define NETWORKS_NETCONFIG_CUH

static dim3 CUDA_BLOCK_SIZE = dim3(32, 32);
static float LEARNING_RATE = 0.05;

static int DATA_SIZE = 50000;
static int BATCH_SIZE = 100;
static int DATA_INPUT_SIZE_1D = 784;
static int DATA_INPUT_SIZE_2D_W = 28;
static int DATA_INPUT_SIZE_2D_H = 28;
static int DATA_OUTPUT_SIZE_1D = 10;

static bool START_FROM_RANDOM = true;
class NetConfig {
public:
};


#endif //NETWORKS_NETCONFIG_CUH
