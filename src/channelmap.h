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
    void InitializeChannelParametrs(unsigned int);
    std::vector<double> Forward(const std::vector<Maps>&);
    std::vector<double> Forward(const std::vector<Maps>&,int);
    

private:
    std::vector<Filters> filters;
    Maps map;
};

#endif