#ifndef FILTERS_H
#define FILTERS_H

#pragma once
#include <vector>
#include <math.h>
#include <Windows.h>
#include <random>

class Filters
{
public:
    Filters();
    ~Filters();
    void SetSize(int height,int width);

    double GetElement(int x,int y) const;
    std::vector<double> Slide(const std::vector<double> &matrix,int stride);

    std::vector<double> SlideRGB(const std::vector<double> &R,const std::vector<double> &G,const std::vector<double> &B,int stride);

    void FillFilter();

    double RandValue(double leftBorder,double rightBorder);

private:
    std::vector<double> filter;
    int width;
    int height;
    double bias = 1.0;

};

#endif