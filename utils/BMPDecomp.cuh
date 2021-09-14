//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_BMPDECOMP_CUH
#define CUDANETWORKS_BMPDECOMP_CUH


#include "MatrixOperations.cuh"

class BMPDecomp {
public:
    static MatrixOperations::Matrix2d* ReadBMP(const string& path);
};



#endif //CUDANETWORKS_BMPDECOMP_CUH
