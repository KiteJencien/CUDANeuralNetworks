//
// Created by DanielSun on 9/14/2021.
//

#include "BMPDecomp.cuh"

__global__ void generateInputMatrix(MatrixOperations::Matrix2d* target, BYTE* file,
                                    BMPDecomp::BITMAPINFOHEADER* bih, int lineByteWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int add =bih->biBitCount / 8;
    BYTE* ptr = file + col * add + (bih->biHeight - 1 - row) * lineByteWidth;
    int rgb = (int)*ptr + (int)*(ptr+1) + (int)*(ptr + 2);
    MatrixOperations::setElement2D(target, row, col, static_cast<float>(rgb) / (256.0F * 3.0F));
}

bool BMPDecomp::ReadBmp(MatrixOperations::Matrix2d *temp, const char *szFileName) {
    BYTE* buffer;
    BITMAPINFOHEADER bih;
    long lineByteWidth;
    FILE *file;
    WORD bfh[7];
    long dpixeladd;

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

    dim3 gridSize = dim3((temp->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (temp->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    generateInputMatrix<<<gridSize, CUDA_BLOCK_SIZE>>>(temp, buffer, &bih, lineByteWidth);
    cudaFree(buffer);
    return true;
}
