#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#pragma once
#include "channals.h"
#include "image.h"

class ConvolutionalLayer
{
public:
    ConvolutionalLayer();
    ~ConvolutionalLayer();

    void SetChannalsNum(int);

    void SetChannalsSize(int);

    std::vector<double> Forward();

private:
    int channalNum;
    int channalsSize;
    std::vector<Channals> channals;
    std::vector<double> img;
    
};

#endif