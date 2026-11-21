#ifndef CHANNEL_H
#define CHANNEL_H

#pragma once
#include "channelmap.h"

class Channel
{
public:
    Channel();
    ~Channel();

    void SetImageMatrix(const std::vector<double>&);

    std::vector<double> GetImageMatrix() const;

    std::vector<Maps> Forward(const std::vector<Maps> &);

    void SetSize(int size);
private:
    std::vector<ChannelMap> channelMaps;
    std::vector<double> imgMatrix; 
    int size = 1;
};

#endif