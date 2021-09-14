cmake_minimum_required(VERSION 3.19)
project(Networks CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories(SYSTEM "/foo/bar")

add_executable(Networks main.cu ../CudaNetworks/utils/MatrixOperations.cu ../CudaNetworks/utils/MatrixOperations.cuh ../CudaNetworks/utils/LogUtils.cu ../CudaNetworks/utils/LogUtils.cuh
        ../CudaNetworks/layers/InputLayerDense.cu ../CudaNetworks/layers/InputLayerDense.cuh ../CudaNetworks/layers/DenseLayer.cu ../CudaNetworks/layers/DenseLayer.cuh NetConfig.cu NetConfig.cuh
        ../CudaNetworks/models/ModelMLP.cu ../CudaNetworks/models/ModelMLP.cuh ../CudaNetworks/layers/Layer.cu ../CudaNetworks/layers/Layer.cuh ../CudaNetworks/dataset/TrainData.cu ../CudaNetworks/dataset/TrainData.cuh
        ../CudaNetworks/dataset/Dataset.cu ../CudaNetworks/dataset/Dataset.cuh ../CudaNetworks/utils/BMPDecomp.cu ../CudaNetworks/utils/BMPDecomp.cuh)

set_target_properties(
        Networks
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
