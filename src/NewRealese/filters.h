#ifndef FILTERS_H
#define FILTERS_H

#pragma once
#include <vector>
#include <iostream> 
#include <math.h>
#include <Windows.h>
class Filters
{
public:
    Filters();
    ~Filters();
    void SetSize(int height,int width);

    double GetElement(int x,int y) const;
    std::vector<double> Slide(const std::vector<double> &matrix,int stride);

private:
    std::vector<double> filter;
    int width;
    int height;
    

    //double& operator[](int x,int y);
};

#endif