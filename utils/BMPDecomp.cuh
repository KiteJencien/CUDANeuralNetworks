//
// Created by DanielSun on 9/14/2021.
//

#ifndef NETWORKS_BMPDECOMP_CUH
#define NETWORKS_BMPDECOMP_CUH

#include "MatrixOperations.cuh"

class BMPDecomp {
public:
    static MatrixOperations::Matrix2d* ReadBMP(const string& path);
};


#endif //NETWORKS_BMPDECOMP_CUH
