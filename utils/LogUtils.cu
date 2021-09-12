//
// Created by DanielSun on 8/30/2021.
//

#include "LogUtils.cuh"
#include <windows.h>

std::string LogUtils::generatePrefix() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer)," %H:%M:%S ",timeinfo);
    std::string str(buffer);

    string out = "[";
    out.append(str).append(" pid:").append(to_string(_getpid())).append(" ] ");
    return out;
}

void LogUtils::logInfo( const string &info) {

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 0x02);
    printf("%s | I >>> %s \n", generatePrefix().c_str(), info.c_str());

}

void LogUtils::logInfo(const char *chars) {
    logInfo(string(chars));
}

void LogUtils::logErr(const string &info) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 0x04);
    printf("%s | E >>> %s \n" , generatePrefix().c_str(), info.c_str());
}

void LogUtils::logErr(const char *chars) {
    logErr(string(chars));
}

void LogUtils::logInfo( const string &info, short color) {

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
    printf("%s | I >>> %s \n", generatePrefix().c_str(), info.c_str());

}

