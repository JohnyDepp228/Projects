#ifndef MAPOFSIGNS_H
#define MAPOFSIGNS_H

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iterator>
#include <string>
#include <iostream>


class MapOfSigns
{
public:
    MapOfSigns();
    ~MapOfSigns();

    std::vector<double> GetMap() const;
    std::vector<double> GetMapBeforeFun() const;
    std::vector<int> GetPoolingMaxIdx() const;
    std::vector<double> GetErrorMap() const;
    std::vector<double> GetPrevLayerError() const;

     
    void SetMap(const std::vector<double> &map);
    void SetMapBeforeFun(const std::vector<double> &mapBeforeFun);
    void SetErrorMap(const std::vector<double> &errorMap);
    void SetPrevLayerError(const std::vector<double> &preLayerError);
    void SetOriginalInput(const std::vector<double> &input);

    std::vector<double> Pooling(const std::vector<double> &matrix);
    std::vector<double> ReversePooling(int targetSize);

    void ApplyActivFun( std::vector<double> &matrix);
    
    void ApplyDiractiveActivFun(std::vector<double> &errorMat);

    double GeLu(double data);

    double DiractiveGeLu(double data);

    double GMP();

    int GetGMPIDx() const;

    std::vector<double> MapsSlide(const std::vector<double> &errorMap);


private:
    int height;
    int width;
    int poolingWidth = 2;
    int poolingHeight = 2;
    int poolingStride = 2;

    int GMPIdx;
    std::vector<double> errorMat;
    std::vector<double> originalInput;
    std::vector<double> mapOfSigns;
    std::vector<double> mapOfSignsBeforeActiveFunction;

    std::vector<double> PrevLayerError;
    std::vector<int> poolingMaxNumIdx;
    const double PI = 3.141592653589793;
    double bias = 0.01;
};

#endif