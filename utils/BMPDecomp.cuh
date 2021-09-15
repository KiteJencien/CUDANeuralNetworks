//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_BMPDECOMP_CUH
#define CUDANETWORKS_BMPDECOMP_CUH


#include "MatrixOperations.cuh"
#include "../NetConfig.cuh"
#include <fstream>

//these bytes are the header section that does not contain pixel infos
const static int BMP_HEADER_OFFSET = 54;

class BMPDecomp {
public:
    typedef unsigned long       DWORD;
    typedef unsigned char       BYTE;
    typedef unsigned short      WORD;

    struct BITMAPINFOHEADER {
        DWORD	biSize;
        long	biWidth;
        long	biHeight;
        WORD	biPlanes;
        WORD	biBitCount;
        DWORD	biCompression;
        DWORD	biSizeImage;
        long	biXPelsPerMeter;
        long	biYPelsPerMeter;
        DWORD	biClrUsed;
        DWORD	biClrImportant;
    };

    static bool ReadBmp(MatrixOperations::Matrix2d *temp, const char *szFileName);
};



#endif //CUDANETWORKS_BMPDECOMP_CUH
