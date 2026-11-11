#ifndef READANDSHUFFLE_H
#define READANDSHUFFLE_H

#pragma once
#include <vector>
#include <filesystem>
#include <string> 
#include <algorithm>
#include <random>
#include <cstdlib>

struct DatasetItem{
    int targetAnswer;
    std::string path;
};

class ReadAndShuffle
{
public:
    ReadAndShuffle();
    ~ReadAndShuffle();

    void ReadAndShuffleTrain(std::vector<DatasetItem> &,std::string);
  

private:
    std::string datasetPathTrain = "../Pupa/dataset/fish-vista_split/train";
    std::string datasetPathTest  = "../Pupa/dataset/fish-vista_split/test";
};

#endif