cmake_minimum_required(VERSION 3.19)
project(Networks CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories(SYSTEM "/foo/bar")

add_executable(Networks main.cu utils/MatrixOperations.cu utils/MatrixOperations.cuh utils/LogUtils.cu utils/LogUtils.cuh layers/InputLayerDense.cu layers/InputLayerDense.cuh layers/DenseLayer.cu layers/DenseLayer.cuh NetConfig.cu NetConfig.cuh models/ModelMLP.cu models/ModelMLP.cuh layers/Layer.cu layers/Layer.cuh dataset/TrainData.cu dataset/TrainData.cuh)

set_target_properties(
        Networks
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)