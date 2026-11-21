#include "channelmap.h"
#include "maps.h"
#include <string>


ChannelMap::ChannelMap()
{

}

ChannelMap::~ChannelMap()
{

}
void ChannelMap::InitializeChannelParametrs(unsigned int filtersNumber) {
    filters.resize(filtersNumber);
    unsigned int filterHeight  = 3;
    unsigned int filterWidth = 3;
    unsigned int filterStride = 1;
    unsigned int poolignStride  = 2;
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
    if(filters.size() != images.size()){
        throw std::string("Filters size didn't match with maps count");
    }
    std::vector<double> sum;
        for(int i = 0;i < images.size();i++){
            if(images[i].GetMapOfSigns().size() < filters[i].GetFilter().size()){
                throw std::string("Image matrix size is too small:\t" + std::to_string(images[i].GetMapOfSigns().size()) + 
                "\nWhen filter size is:\t" + std::to_string( filters[i].GetFilter().size()));
            }
            std::vector<double> temp = filters[i].Slide(images[i].GetMapOfSigns(),filters[i].GetFilter(),1);
            if(i == 0){
                sum = temp;
            }
            else{
                sum = sum + temp;
            }
        }
    
    std::vector<double> mapTempRes = map.ApplyActivateFunction(sum);
    std::vector<double> res = map.Pooling(mapTempRes);
    map.SetMap(res);
    return this->map.GetMapOfSigns();
}
