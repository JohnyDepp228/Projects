#ifndef MAPS_H
#define MAPS_H

#pragma once
#include <vector> 
#include <string> 
#include <cmath>

class Maps
{
public:
    Maps();
    ~Maps();

    std::vector<double> Pooling(const std::vector<double> &mapsOfSigns);

    std::vector<double> ApplyActivateFunction(const std::vector<double> &mapOfSigns);

    void ApplyDiractivateFunction();

    std::vector<double> ReversePooling(const std::vector<double> &mapOfErrors);

    double GeLu(double);

    double DiractiveGeLu(double);

    double GMP();

    std::vector<double> operator* (const std::vector<double> &reversePoolingMap) const;

    void FindMapForLearning(const std::vector<double> &mapOfErrors);

    //Setters 
    void SetHeight(unsigned int height);

    void SetWidth(unsigned int width);

    void SetMap(const std::vector<double> &mapOfSigns);

    void SetStride(int stride);

    void SetMapBeforeActivate(const std::vector<double> &map);

    
    //Getters

    unsigned int GetHeight() const;

    unsigned int GetWidth() const;

    std::vector<double> GetMapOfSigns() const;

    std::vector<double> GetMapOfErros() const;

    int GetStride() const;

    std::vector<int> GetPooledIdx() const;

    int GetGMPIdx() const;

private:
    std::vector<double> mapOfsigns;
    std::vector<double> mapBeforeActivateFunction;
    std::vector<int> pooledValuesIdx;

    std::vector<double> mapOfErrors;
    const double PI = 3.141592653589793;
    int stride = 2;
    const int poolHeight = 2;
    const int poolWidth = 2;
    int GMPIdx = 0;
    unsigned int mapWidth = 0;
    unsigned int mapHeight = 0;
   
};

#endif