#ifndef FILTERS_H
#define FILTERS_H

#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <random>

class Filters
{
public:
    Filters();
    ~Filters();

    void FillFilter();

    double RandValue(double leftBorder,double rightBorder);

    bool HasNanOrInf(const std::vector<double> &vec);

    std::vector<double> Slide(const std::vector<double> &bigMatrix,const std::vector<double> &smallMatrix);

    //Setters

    void SetHeight(unsigned int height);

    void SetWidth(unsigned int width);

    void SetFilter(const std::vector<double> &filter);

    void SetStride(int stride);

    //Getters

    unsigned int GetHeight() const;

    unsigned int GetWidth() const;

    std::vector<double> GetFilter() const;

    int GetStride() const;


private:
    unsigned int height = 1;
    unsigned int width = 1;
    std::vector<double> filter;
    int stride = 1;
    double bias = 0.1;
};

#endif