#ifndef CONFIG_H
#define CONFIG_H

#pragma once
#include <fstream> 
#include <string> 
#include <vector>
#include <cerrno>

class Config
{
public:
    Config();
    ~Config();
    void SetPath(const std::string &path);

    bool SaveData(std::string dataType, int data);

    bool SaveData(std::string dataType,const std::vector<double> &data);

    void ReadData(std::string dataType, int &data);

    void ReadData(std::string dataType,std::vector<double> &data);


private:
    std::string path = "./config.txt";
};

#endif