#ifndef FILTERS_H
#define FILTERS_H

#pragma once
#include <vector>
#include <iostream> 
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

    void FillFilter();

    double RandValue(double leftBorder,double rightBorder);

private:
    std::vector<double> filter;
    int width;
    int height;

};

#endif