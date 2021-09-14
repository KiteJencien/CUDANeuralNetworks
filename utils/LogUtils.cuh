//
// Created by DanielSun on 8/30/2021.
//

#ifndef NETWORKS_LOGUTILS_CUH
#define NETWORKS_LOGUTILS_CUH

#include "string"
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


#endif //NETWORKS_LOGUTILS_CUH
