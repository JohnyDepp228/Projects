#include "channals.h"
#include "config.h"
#include <string>

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
        std::vector<double> temp;
        temp = filters[idx].Slide(mat,this->convStride);
        maps[idx].SetMapBeforeFun(temp);
        maps[idx].SetOriginalInput(this->inputMat);
        maps[idx].ApplyActivFun(temp);
        mat = maps[idx].Pooling(temp);
        maps[idx].SetMap(mat);
       
    }
}

std::vector<double>  Channals::ConvResGray(const std::vector<double> &gray) {
    std::vector<double> temp;
    std::vector<double> temp2;
    SetInput(gray);
    for(int i =0;i < this->channalSize;i++){
        temp = filters[i].Slide(gray,this->convStride);
      
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
        const std::vector<double>& currentMap = maps[i].GetMap(); 
        for(int j =0; j< matSize;j++){
            res[j] += currentMap[j]; 
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
    this->inputMat = input;
    int finalSize;
    std::vector<double> currentInput; 
    for(int i =0;i < this->channalSize;i++){
        currentInput = input; 
        ConvBlock(i,currentInput);
        if(i ==0){
            finalSize = currentInput.size();
        }
    }
   
    res = ChannalSum(finalSize);
   
    return res;
}

std::vector<double> Channals::ChannalGMP(const std::vector<double> &matrix) {
    std::vector<double> temp;
    for(int i =0;i < filters.size();i++){
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
   
}

void Channals::ChannalLearnGMP(const std::vector<double> &classifierError) {
    ReversGMP(classifierError);
    std::vector<double> temp;
    std::vector<double> temp1;
    for(int i =0;i < channalSize;i++){
        maps[i].SetOriginalInput(this->inputMat);
      
        temp = MultiMapWithErrorMap(i, maps[i].GetErrorMap());
        
        maps[i].ApplyDiractiveActivFun(temp);
       
        temp1 = maps[i].MapsSlide(temp);
       
        filters[i].LearnFilters(temp1);
        
        maps[i].SetPrevLayerError(filters[i].SlideWithReversed(temp,1));
    }
}

void Channals::ChannalLearn() {
    std::vector<double> unpooledMap;
    std::vector<double> temp;
    std::vector<double> temp1;
    std::vector<double> prevLayerError;
    for(int i = 0; i < channalSize; i++){
        int targetSize = maps[i].GetMapBeforeFun().size();

        unpooledMap =  maps[i].ReversePooling(targetSize);
       
        maps[i].SetOriginalInput(this->inputMat);

        temp = MultiMapWithErrorMap(i, unpooledMap);

        for(int j =0;j < temp.size();j++){
            if(!std::isfinite(temp[j])){
                std::cout << "Invalid Before GeLu in map\t" << i << "\tIndex:\t" << j << "\tValue\t" << temp[j] << std::endl; 
                throw std::string("Before Gelu");
            }
        }
       
        maps[i].ApplyDiractiveActivFun(temp);

        for(int j =0;j < temp.size();j++){
            if(!std::isfinite(temp[j])){
                std::cout << "Invalid After GeLu in map\t" << i << "\tIndex:\t" << j << "\tValue\t" << temp[j] << std::endl; 
                throw std::string("After Gelu");
            }
        }
        
        temp1 = maps[i].MapsSlide(temp);
       
        filters[i].LearnFilters(temp1);
        
        prevLayerError = filters[i].SlideWithReversed(temp, 1);
        
        if (prevLayerError.empty()) {
            throw std::string("SlideWithReversed returned empty vector for map " + std::to_string(i));
        }
        
        maps[i].SetPrevLayerError(prevLayerError);
    }
}




std::vector<double> Channals::MultiMapWithErrorMap(int mapIdx,const std::vector<double> &errorMap) {
    std::vector<double> res = errorMap;
    if (res.size() != errorMap.size()) {
        throw std::string("Size mismatch in MultiMapWithErrorMap: res is " + 
                          std::to_string(res.size()) + ", but errorMap is " + std::to_string(errorMap.size()) + "\n");
    }
    for(int i =0;i < res.size();i++){
        if (!std::isfinite(res[i])) {
        throw std::string("res[i] is invalid before multiplication!\n");
        }
        if (!std::isfinite(res[i])) {
            throw std::string("Res is invalid before multiplication at index " + std::to_string(i)+ "\n");
        }
        if(!std::isfinite(errorMap[i])){
            throw std::string("invalid data when map multi in error map " + std::to_string(i) + "\n");
        } 
       // res[i] = errorMap[i];
        if (!std::isfinite(res[i])){
            throw std::string("invalid data when map multi in res map " + std::to_string(i) + "\t" + std::to_string(errorMap[i]) + "\n");
        } 
    }

    return res;
}

void Channals::SetInput(const std::vector<double> &input) {
    this->inputMat = input;
}

std::vector<double> Channals::GetInput()const {
    return this->inputMat;
}

void Channals::SetErrorMap(int mapIdx,const std::vector<double> &error) {
    maps[mapIdx].SetErrorMap(error);
}

std::vector<double> Channals::GetErrorMap(int mapIdx)const {
    return maps[mapIdx].GetErrorMap();    
}

int Channals::GetChannelSize() const {
    return this->channalSize;
}

std::vector<double> Channals::GetPrevLayerError(int mapIdx) {
    return maps[mapIdx].GetPrevLayerError();
}

bool Channals::HasNaN(const std::vector<double> &vec) {
    for(int i = 0;i < vec.size();i++){
        if (!std::isfinite(vec[i])) return true;
    }
    return false;
}

void Channals::SaveFiltersConfig(std::string path,int channelIdx) {
    Config conf;
    conf.SetPath(path);
    std::string dataReference = "ChannelFilter" + std::to_string(channelIdx);
    conf.SaveData(dataReference + "Size", filters.size());
    conf.SaveData(dataReference + "Height", filters[0].GetHeight());
    conf.SaveData(dataReference + "Width", filters[0].GetWidth());
    for(int i =0;i < filters.size();i++){
        conf.SaveData(dataReference + "Idx" + std::to_string(i), filters[i].GetFilter());
    }

}

void Channals::ReadFiltersConfig(std::string path,int channelIdx) {
    Config conf;
    conf.SetPath(path);
    std::string dataReference = "ChannelFilter" + std::to_string(channelIdx);
    int filtersSize = -1;
    int filterWidth = -1;
    int filterHeight = -1;
    conf.ReadData(dataReference + "Size", filtersSize);
    conf.ReadData(dataReference + "Height", filterHeight);
    conf.ReadData(dataReference + "Width", filterWidth);
    SetFiltersParams(filtersSize, filterHeight, filterWidth);
    for(int i =0;i < filtersSize;i++){
        std::vector<double> filter;
        conf.ReadData(dataReference + "Idx" + std::to_string(i), filter);
        filters[i].SetFilter(filter);
    }   
}

void Channals::Clean() {
    for(int i =0;i < channalSize;i++){
        maps[i].Reset();
    }
}




