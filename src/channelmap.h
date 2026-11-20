#ifndef CHANNELMAP_H
#define CHANNELMAP_H

#pragma once
#include "filters.h"
#include "maps.h"

class ChannelMap
{
public:
    ChannelMap();
    ~ChannelMap();
    void InitializeChannelParametrs(unsigned int filterHeight,unsigned int filterWidth,
                             unsigned int filterStride,unsigned int filtersNumber,
                             unsigned int poolignStride);
    std::vector<double> Forward(const std::vector<Maps>&);

    

private:
    std::vector<Filters> filters;
    Maps map;
};

#endif