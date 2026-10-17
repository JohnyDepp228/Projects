#ifndef MAPOFSIGNS_H
#define MAPOFSIGNS_H

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iterator>


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

    void ApplyActivFun( std::vector<double> &matrix);
    
    void ApplyDiractiveActivFun();

    double GeLu(double data);

    double DiractiveGeLu(double data);

    double GMP();


private:
    int height;
    int width;
    int poolingWidth = 2;
    int poolingHeight = 2;
    int poolingStride = 2;

    int GMPIdx;
    std::vector<double> mapOfSigns;
    std::vector<double> mapOfSignsBeforeActiveFunction;
    std::vector<int> poolingMaxNumIdx;
    const double PI = 3.141592653589793;
    double bias = 1.0;
};

#endif