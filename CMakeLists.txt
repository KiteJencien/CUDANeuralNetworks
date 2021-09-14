cmake_minimum_required(VERSION 3.19)
project(CudaNetworks CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaNetworks main.cu NetConfig.cu NetConfig.cuh utils/LogUtils.cu utils/LogUtils.cuh utils/MatrixOperations.cu utils/MatrixOperations.cuh utils/BMPDecomp.cu utils/BMPDecomp.cuh layers/DenseLayer.cu layers/DenseLayer.cuh layers/Layer.cu layers/Layer.cuh layers/InputLayerDense.cu layers/InputLayerDense.cuh models/ModelMLP.cu models/ModelMLP.cuh datasets/TrainData.cu datasets/TrainData.cuh datasets/Dataset.cu datasets/Dataset.cuh)

set_target_properties(
        CudaNetworks
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)