#ifndef FILTERS_H
#define FILTERS_H

#pragma once
#include <vector>
#include <math.h>
#include <Windows.h>
#include <random>
#include <algorithm>

class Filters
{
public:
    Filters();
    ~Filters();
    void SetSize(int height,int width);

    double GetElement(int x,int y) const;

     double GetElementFromReversed(int x,int y) const;

    std::vector<double> Slide(const std::vector<double> &matrix,int stride);

     std::vector<double> SlideWithReversed(const std::vector<double> &matrix,int stride);

    std::vector<double> SlideRGB(const std::vector<double> &R,const std::vector<double> &G,const std::vector<double> &B,int stride);

    void FillFilter();

    double RandValue(double leftBorder,double rightBorder);

    void LearnFilters(const std::vector<double> &errorMat);

    std::vector<double> GetReverseFilter() const;

private:
    std::vector<double> filter;
    std::vector<double> revFilter;
    int width;
    int height;
    double bias = 1.0;
    double LR = 0.5;

};

#endif