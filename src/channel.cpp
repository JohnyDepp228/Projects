#include "channel.h"

Channel::Channel()
{

}

Channel::~Channel()
{

}
void Channel::SetImageMatrix(const std::vector<double> &imgMatrix) {
    this->imgMatrix = imgMatrix;
}
std::vector<double> Channel::GetImageMatrix() const {
    return this->imgMatrix;
}

std::vector<Maps> Channel::Forward(const std::vector<Maps> &images) {
    if(size <= 0){
        throw std::string("Invalid channel size");
    }
    for(int i =0;i < size;i++){
        channelMaps[i].InitializeChannelParametrs(images.size());
    }
    
    std::vector<Maps> res(size);
    std::vector<double> temp;
    for(int i =0;i < size;i++){
        temp = channelMaps[i].Forward(images);
        res[i].SetMap(temp);
    }

    return res;
}

void Channel::SetSize(int size) {
    this->size = size;
    channelMaps.resize(size);
}
