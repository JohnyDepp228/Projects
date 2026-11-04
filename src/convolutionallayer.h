#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#pragma once
#include "channals.h"
#include "image.h"
#include <iostream>
#include <string>
#include "config.h"

class ConvolutionalLayer
{
public:
    ConvolutionalLayer();
    ~ConvolutionalLayer();

    void SetChannalsNum(int);

    void SetChannalsSize(int);

    std::vector<double> Forward();

    void Backpropageation(const std::vector<double> &classifierError);

    void SetPrevLayerErrorMap(int currentChannelIdx,int prevChannelIdx);

    void SetImage(std::string path);

    void SetChannelsParams();

    void SetConfigPath(std::string path);

    void SaveChannelConfig();

    void ReadChannelConfig();

private:
    int channalNum;
    int channalsSize;
    std::vector<Channals> channals;
    std::vector<double> img;
    Config conf;
    std::string path;
};

#endif