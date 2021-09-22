//
// Created by DanielSun on 9/14/2021.
//

#include "BMPDecomp.cuh"

__global__ void generateInputMatrix(MatrixOperations::Matrix2d* target, BYTE* buffer,
                                   int biHeight, int biBitCount, int lineByteWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int add = biBitCount / 8;
    BYTE* ptr = buffer + col * add + (biHeight - 1 - row) * lineByteWidth;
    int b = *ptr;
    int g = *(ptr + 1);
    int r = *(ptr + 2);
    MatrixOperations::setElement2D(target, row, col, static_cast<float>(r+g+b) / (256.0F * 3.0F));
}

__global__ void forceGenerateInputMatrix(MatrixOperations::Matrix2d* target, unsigned char* buffer){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (row*target->colcount) + col;
    unsigned char r = buffer[54 + index*4];
    unsigned char g = buffer[54 + index*4+1];
    unsigned char b = buffer[54 + index*4+2];
    MatrixOperations::setElement2D(target, row, col, static_cast<float>((int)r+(int)g+(int)b) / (256.0F * 3.0F));
}

bool BMPDecomp::ReadBmp(MatrixOperations::Matrix2d *temp, const char *szFileName) {
    BYTE* buffer;
    BITMAPINFOHEADER bih{};
    long lineByteWidth;
    FILE *file;
    WORD bfh[7];
    long dpixeladd;
    MatrixOperations::Matrix2d* calcTemp;
    cudaMallocManaged(reinterpret_cast<void**>(&calcTemp), sizeof(MatrixOperations::Matrix2d));
    MatrixOperations::callAllocElementD(calcTemp, temp->rowcount, temp->colcount);

    if(nullptr == (file = fopen(szFileName, "rb"))) {
        return false;
    }

    fread(&bfh, sizeof(WORD), 7, file);
    if(bfh[0] != (WORD)(((WORD)'B')|('M'<<8))) {
        fclose(file);
        LogUtils::logErr("File : " + string(szFileName) + " is not bmp format");
        return false;
    }

    fread(&bih, sizeof(BITMAPINFOHEADER), 1, file);

    if(bih.biBitCount != 24) {
        fclose(file);
        LogUtils::logErr("File : " + string(szFileName) + " is not 24 bit bmp");
        return false;
    }

    dpixeladd = bih.biBitCount / 8;
    lineByteWidth = bih.biWidth * (dpixeladd);
    cudaMallocManaged(reinterpret_cast<void**>(&buffer), sizeof(BYTE)*lineByteWidth*bih.biHeight);
    if((lineByteWidth % 4) != 0) {
        lineByteWidth += 4 - (lineByteWidth % 4);
    }
    fread(buffer, lineByteWidth * bih.biHeight, 1, file);
    fclose(file);

    if(bih.biHeight != MNIST_INPUT_DIMENSION_ROW || bih.biWidth != MNIST_INPUT_DIMENSION_COL){
        LogUtils::logErr("Input Size Mismatch");
        return false;
    }

    dim3 gridSize = dim3((calcTemp->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (calcTemp->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    generateInputMatrix<<<gridSize, CUDA_BLOCK_SIZE>>>(calcTemp, buffer, bih.biHeight, bih.biBitCount, lineByteWidth);
    (void) cudaDeviceSynchronize();
    MatrixOperations::inspect(calcTemp);
    MatrixOperations::callMatCopyD2H(calcTemp,temp);
    cudaFree(calcTemp->elements);
    cudaFree(calcTemp);
    cudaFree(buffer);
    return true;
}

bool BMPDecomp::forceReadBMP(MatrixOperations::Matrix2d *temp, const char *szFileName) {
    FILE* file;
    unsigned char* buffer;
    MatrixOperations::Matrix2d* calcTemp;

    cudaMallocManaged(reinterpret_cast<void**>(&calcTemp), sizeof(MatrixOperations::Matrix2d));
    MatrixOperations::callAllocElementD(calcTemp, temp->rowcount, temp->colcount);
    cudaMallocManaged(reinterpret_cast<void**>(&buffer), sizeof (DATA_INPUT_SIZE_1D*4) + 54);
    if(nullptr == (file = fopen(szFileName, "rb"))) {
        return false;
    }
    fread(buffer,DATA_INPUT_SIZE_1D*4+54,1,file);
    fclose(file);
    dim3 gridSize = dim3((calcTemp->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (calcTemp->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    forceGenerateInputMatrix<<<gridSize, CUDA_BLOCK_SIZE>>>(calcTemp, buffer);
    (void) cudaDeviceSynchronize();
    MatrixOperations::callMatCopyD2H(calcTemp,temp);
    cudaFree(calcTemp->elements);
    cudaFree(calcTemp);
    cudaFree(buffer);
    return true;
}
