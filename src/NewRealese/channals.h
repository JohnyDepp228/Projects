#ifndef CHANNALS_H
#define CHANNALS_H

#pragma once
#include <vector> 
#include <string>
#include "filters.h"
#include "mapofsigns.h"


class Channals
{
public:
    Channals();
    ~Channals();

    void SetFiltersParams(int amount,int height,int width);

    void ConvBlock(int idx,std::vector<double> &mat);

    std::vector<double>  ConvResRGB(const std::vector<double> &R,const std::vector<double> &G,const std::vector<double> &B);
    
    std::vector<double> ChannalSum(int matSize);

    void SetChannalSize(int size);

    std::vector<double> ChannalRes(const std::vector<double> &input);

    std::vector<double> ChannalGMP(const std::vector<double> &matrix);

private:
    std::vector<MapOfSigns> maps;
    std::vector<Filters> filters;
    std::string imgPath;
    const int convStride = 1;
    int channalSize = 1;
};

#endif