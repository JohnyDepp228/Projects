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

    //Setters 
    void SetHeight(unsigned int height);

    void SetWidth(unsigned int width);

    void SetMap(const std::vector<double> &mapOfSigns);

    void SetStride(int stride);

    //Getters

    unsigned int GetHeight() const;

    unsigned int GetWidth() const;

    std::vector<double> GetMapOfSigns() const;

    int GetStride() const;

    std::vector<int> GetPooledIdx() const;

private:
    std::vector<double> mapOfsigns;
    unsigned int mapWidth = 0;
    unsigned int mapHeight = 0;
    int stride = 2;
    int poolHeight = 2;
    int poolWidth = 2;
    std::vector<int> pooledValuesIdx;
};

#endif