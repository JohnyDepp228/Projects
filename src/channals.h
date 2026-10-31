#ifndef CHANNALS_H
#define CHANNALS_H

#pragma once
#include <vector> 
#include <string>
#include "filters.h"
#include <fstream>
#include <cmath>
#include "mapofsigns.h"


class Channals
{
public:
    Channals();
    ~Channals();

    void SetFiltersParams(int amount,int height,int width);

    void ConvBlock(int idx,std::vector<double> &mat);

    std::vector<double>  ConvResGray(const std::vector<double> &gray);
    
    std::vector<double> ChannalSum(int matSize);

    void SetChannalSize(int size);

    std::vector<double> ChannalRes(const std::vector<double> &input);

    std::vector<double> ChannalGMP(const std::vector<double> &matrix);

    void ReversGMP(const std::vector<double> &classifierError);

    void ChannalLearnGMP(const std::vector<double> &classifierError);

    void ChannalLearn();

    std::vector<double> MultiMapWithErrorMap(int mapIdx,const std::vector<double> &errorMap);

    void SetInput(const std::vector<double> &input);

    std::vector<double> GetInput()const;

    void SetErrorMap(int mapIdx,const std::vector<double> &error);

    std::vector<double> GetErrorMap(int mapIdx)const;

    int GetChannelSize() const;

    std::vector<double> GetPrevLayerError(int mapIdx);

    bool HasNaN(const std::vector<double> &vec);

private:
    std::vector<MapOfSigns> maps;
    std::vector<Filters> filters;
    std::string imgPath;
    std::vector<double> inputMat;
    const int convStride = 1;
    int channalSize = 1;
};

#endif