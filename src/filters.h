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

    std::vector<double> Slide(const std::vector<double> &bigMatrix,const std::vector<double> &smallMatrix,bool addBias);

    void LearnFilters(const std::vector<double> &mapOfSigns,const std::vector<double> &errorMap);

    std::vector<double> FindMapForPrevLayer(const std::vector<double> &errorMap);

    std::vector<double> operator-(const std::vector<double> &filtersError);

    std::vector<double> Padding(const std::vector<double> &);

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
    std::vector<double> filter;
    double bias = 0.01;
    double LR = 0.001;
    int stride = 1;
    int padding;
    unsigned int height = 1;
    unsigned int width = 1;
    
    
};    

#endif