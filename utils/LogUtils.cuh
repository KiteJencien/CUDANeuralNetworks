//
// Created by DanielSun on 9/14/2021.
//

#ifndef CUDANETWORKS_LOGUTILS_CUH
#define CUDANETWORKS_LOGUTILS_CUH

#include <string>
#include <cstdio>
#include <process.h>
#include <iostream>
#include <vector>
#include <map>
#include <Windows.h>
#include <ctime>

using namespace std;
class LogUtils {
public:
    /**
      * generate prefix
      * @return prefix
      */
    static string generatePrefix();
    /**
     * log info with string
     * @param info
     */
    static void logInfo( const string& info);
    /**
     * with color
     * @param sector
     * @param info
     * @param color
     */
    static void logInfo( const string& info, short color);
    /**
     * same method but in chars
     * @param chars
     */
    static void logInfo( const char *chars);
    /**
     * log error
     * @param info
     */
    static void logErr(const string& info);
    /**
     * log error
     * @param chars
     */
    static void logErr(const char *chars);


};


#endif //CUDANETWORKS_LOGUTILS_CUH
