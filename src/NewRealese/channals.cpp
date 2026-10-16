#include "channals.h"

Channals::Channals()
{

}

Channals::~Channals()
{

}
void Channals::SetFiltersParams(int amount,int height,int width) {
    this->filters.resize(amount);
    for(auto &f : filters){
        f.SetSize(height,width);
        f.FillFilter();
    }
}

void Channals::ConvBlock(int idx,std::vector<double> &mat) {
    if(idx > this->filters.size() || idx > this->maps.size()){
        throw "Invalid idx of filter/map" + std::to_string(idx);
    }
    else {
        std::vector<double> temp;
        temp = filters[idx].Slide(mat,this->convStride);
        maps[idx].SetMapBeforeFun(temp);
        maps[idx].ApplyActivFun(temp);
        mat = maps[idx].Pooling(temp);
        maps[idx].SetMap(mat);
    }
}

std::vector<double>  Channals::ConvResRGB(const std::vector<double> &R,const std::vector<double> &G,const std::vector<double> &B) {
    std::vector<double> temp;
    std::vector<double> temp2;
    for(int i =0;i < this->channalSize;i++){
        temp = filters[i].SlideRGB(R,G,B,this->convStride);
        std::cout << "Slide\n";
        maps[i].SetMapBeforeFun(temp);
        std::cout << "set\n";
        maps[i].ApplyActivFun(temp);
        std::cout << "active\n";
        std::cout << "Start pooling";
        temp2 = maps[i].Pooling(temp);
        std::cout << "Pooling\n";
        maps[i].SetMap(temp2);
    }

    return ChannalSum(temp2.size());
}



std::vector<double> Channals::ChannalSum(int matSize) {
    std::vector<double> res(matSize,0.0);
    for(int i =0;i < this->channalSize;i++){
        std::vector<double> temp = maps[i].GetMap();
        for(int j =0; j< matSize;j++){
            res[j] += temp[j]; 
        }
    }

    return res;
}

void Channals::SetChannalSize(int size) {
    this->channalSize = size;
    this->maps.resize(size);
}

std::vector<double> Channals::ChannalRes(const std::vector<double> &input) {
    std::vector<double> res;
    int finalSize;
    for(int i =0;i < this->channalSize;i++){
        std::vector<double> currentInput = input; 
        ConvBlock(i,currentInput);
        if(i ==0){
            finalSize = currentInput.size();
        }
    }
    std::cout << "Calcul block of conv\n";
    res = ChannalSum(finalSize);
    std::cout << "res size: " << res.size() << std::endl;
    return res;
}

std::vector<double> Channals::ChannalGMP(const std::vector<double> &matrix) {
    for(int i =0;i < filters.size();i++){
        std::vector<double> temp;
        temp = filters[i].Slide(matrix,this->convStride);
        maps[i].SetMapBeforeFun(temp);
        maps[i].ApplyActivFun(temp);
        maps[i].SetMap(temp);
    }
    std::vector<double> res(this->maps.size());
    for(int i =0;i < this->maps.size();i++){
        res[i] = maps[i].GMP();
    }
    return res;
}




