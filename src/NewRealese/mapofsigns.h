#ifndef MAPOFSIGNS_H
#define MAPOFSIGNS_H

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

class MapOfSigns
{
public:
    MapOfSigns();
    ~MapOfSigns();

    std::vector<double> GetMap() const;
    std::vector<double> GetMapBeforeFun() const;
std::vector<int> GetPoolingMaxIdx() const;

     
    void SetMap(const std::vector<double> &map);
    void SetMapBeforeFun(const std::vector<double> &mapBeforeFun);

    std::vector<double> Pooling(const std::vector<double> &matrix);
    std::vector<double> ReversePooling(const std::vector<double> &errorMatrix);

private:
    int height;
    int width;
    int poolingWidth = 2;
    int poolingHeight = 2;
    int poolingStride = 2;
    std::vector<double> mapOfSigns;
    std::vector<double> mapOfSignsBeforeActiveFunction;
    std::vector<int> poolingMaxNumIdx;
};

#endif