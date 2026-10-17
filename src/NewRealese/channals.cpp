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
        throw (std::string("Invalid idx of filter/map") + std::to_string(idx));
    }
    else {
        std::cout << "Input size: " << this->inputMat.size() << std::endl;
        std::vector<double> temp;
        temp = filters[idx].Slide(mat,this->convStride);
        maps[idx].SetMapBeforeFun(temp);
        maps[idx].SetOriginalInput(this->inputMat);
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
      
        maps[i].SetMapBeforeFun(temp);
       
        maps[i].ApplyActivFun(temp);
      
        temp2 = maps[i].Pooling(temp);
       
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
   
    res = ChannalSum(finalSize);
   
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

void Channals::ReversGMP(const std::vector<double> &classifierError) {
    std::vector<double> res(maps[0].GetMap().size(),0.0);
    for(int i = 0 ;i < channalSize;i++){
        res.assign(maps[i].GetMap().size(), 0.0); 
        int idx =  maps[i].GetGMPIDx();
        res[idx] = classifierError[i];
        maps[i].SetErrorMap(res);
    }
    std::cout << "Reversed map size: " << res.size() << std::endl;
}

void Channals::ChannalLearn() {
    std::cout << "Learning channal size of " << channalSize << std::endl;
    for(int i =0;i < channalSize;i++){
       // std::cout << "Learning idx " << i << std::endl;
        std::vector<double> temp = MultiMapWithErrorMap(i, maps[i].GetErrorMap());
        maps[i].ApplyDiractiveActivFun(temp);
        std::vector<double> temp1;
        std::cout << "Filter slide size: "<<temp.size();
        temp1 = maps[i].MapsSlide(temp);
        std::cout << "Filters map for learn size: " << temp1.size() << std::endl;
        std::cout << "Old filter weight\n";
         for(int i =0;i < 3;i++){
            for(int j = 0;j < 3;j++){
                std::cout << filters[i].GetElement(i, j);
            }
        }
        std::cout << std::endl;
        filters[i].LearnFilters(temp);
        std::cout << "New filter weight\n";
        for(int i =0;i < 3;i++){
            for(int j = 0;j < 3;j++){
                std::cout << filters[i].GetElement(i, j);
            }
        }
        std::cout << std::endl;
        maps[i].SetPrevLayerError(filters[i].SlideWithReversed(temp,1));
    }
}

std::vector<double> Channals::MultiMapWithErrorMap(int mapIdx,const std::vector<double> &errorMap) {
    std::vector<double> res = maps[mapIdx].GetMapBeforeFun();
    for(int i =0;i < res.size();i++){
        res[i] *= errorMap[i];
    }

    return res;
}

void Channals::SetInput(const std::vector<double> &input) {
    this->inputMat = input;
}

std::vector<double> Channals::GetInput()const {
    return this->inputMat;
}




