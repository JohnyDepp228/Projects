#include "channelmap.h"
#include "maps.h"


ChannelMap::ChannelMap()
{

}

ChannelMap::~ChannelMap()
{

}
void ChannelMap::InitializeChannelParametrs(unsigned int filterHeight,unsigned int filterWidth,
                                  unsigned int filterStride,unsigned int filtersNumber,
                                  unsigned int poolignStride) {
    filters.resize(filtersNumber);
    

    for(int i =0;i < filtersNumber;i++){
        filters[i].SetHeight(filterHeight);
        filters[i].SetWidth(filterWidth);
        filters[i].SetStride(filterStride);
        filters[i].FillFilter();
    }
    map.SetStride(poolignStride);
}

std::vector<double> operator+(const std::vector<double>&vec1,std::vector<double>&vec2){
    if(vec1.size() != vec2.size()){
        throw std::string("Invalid sizee of matrix to multi");
    }
    std::vector<double> res(vec2.size(),0.0);
    for(int i =0;i < res.size();i++){
        res[i] = vec1[i] + vec2[i];
    }
    return res;
}
//это функция считает одну карту для слоев 1-n а на первом слое суммы карт не должно быть там только проход
std::vector<double> ChannelMap::Forward(const std::vector<Maps> &images) {
    std::vector<double> sum;
        for(int i = 0;i < images.size();i++){
            std::vector<double> temp = filters[i].Slide(images[i].GetMapOfSigns(),filters[i].GetFilter(),1);
            if(i == 0){
                sum = temp;
            }
            else{
                sum = sum + temp;
            }
        }

        map.SetMap(sum);
        map.ApplyActivateFunction(map.GetMapOfSigns());
        map.Pooling(map.GetMapOfSigns());

    return this->map.GetMapOfSigns();
}
