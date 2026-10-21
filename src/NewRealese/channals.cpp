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
        std::cout << "Map size: " << mat.size() << std::endl;
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
    this->inputMat = input;
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

void Channals::ChannalLearnGMP(const std::vector<double> &classifierError) {
    ReversGMP(classifierError);
    std::cout << "Learning channal size of " << channalSize << std::endl;
    for(int i =0;i < channalSize;i++){
        maps[i].SetOriginalInput(this->inputMat);
        std::cout << "Channal learn set size of input layer: " << this->inputMat.size() << std::endl;
        std::vector<double> temp = MultiMapWithErrorMap(i, maps[i].GetErrorMap());
        maps[i].ApplyDiractiveActivFun(temp);
        std::vector<double> temp1;
        std::cout << "Filter slide size: "<<temp.size() << std::endl;
        temp1 = maps[i].MapsSlide(temp);
        std::cout << "Filters map for learn size: " << temp1.size() << std::endl;
        std::cout << "Old filter weight\n";
         for(int i =0;i < 3;i++){
            for(int j = 0;j < 3;j++){
                std::cout << filters[i].GetElement(i, j);
            }
        }
        std::cout << std::endl;
        filters[i].LearnFilters(temp1);
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

void Channals::ChannalLearn() {
    std::cout << "Start channal learning" << std::endl;
    for(int i = 0; i < channalSize; i++){
        int targetSize = maps[i].GetMapBeforeFun().size();
       std::vector<double> unpooledMap =  maps[i].ReversePooling(targetSize);
        std::cout << "Reverse pooling done in idx " << i << std::endl;
        maps[i].SetOriginalInput(this->inputMat);
        
        std::vector<double> currentErrorMap = maps[i].GetErrorMap();
        std::vector<double> temp = MultiMapWithErrorMap(i, unpooledMap);
        std::cout << "Multi map with error pooling done in idx " << i << std::endl;
         // ДЕБАГ 1: Проверяем ошибку до активации
        double sum1 = 0; for(double v : temp) sum1 += std::abs(v);
        std::cout << "Sum after MultiMap: " << sum1 << std::endl;


        maps[i].ApplyDiractiveActivFun(temp);
        std::vector<double> temp1;
        std::cout << "Filter slide size: " << temp.size() << std::endl;
        // ДЕБАГ 2: Проверяем ошибку ПОСЛЕ производной активации (Подозреваемый №1)
        double sum2 = 0; for(double v : temp) sum2 += std::abs(v);
        std::cout << "Sum after ActivFun Derivative: " << sum2 << std::endl;
        temp1 = maps[i].MapsSlide(temp);
        // ДЕБАГ 3: Проверяем градиент весов
        double sum3 = 0; for(double v : temp1) sum3 += std::abs(v);
        std::cout << "Sum of weight gradients (temp1): " << sum3 << std::endl;
        std::cout << "Map slide done in idx " << i << std::endl;
        std::cout << "Filters map for learn size: " << temp1.size() << std::endl;
        
        std::cout << "Old filter weight\n";
        for(int k = 0; k < 3; k++){
            for(int j = 0; j < 3; j++){
                std::cout << filters[i].GetElement(k, j); 
            }
        }
        std::cout << std::endl;
        
        filters[i].LearnFilters(temp1);
        std::cout << "Learn learning done in idx " << i << std::endl;
        
        std::cout << "New filter weight\n";
        for(int k = 0; k < 3; k++){
            for(int j = 0; j < 3; j++){
                std::cout << filters[i].GetElement(k, j);
            }
        }
        std::cout << std::endl;
        
      
        std::vector<double> prevLayerError = filters[i].SlideWithReversed(temp, 1);
        
       
        if (prevLayerError.empty()) {
            throw std::string("SlideWithReversed returned empty vector for map " + std::to_string(i));
        }
        
        maps[i].SetPrevLayerError(prevLayerError);
    }
}




std::vector<double> Channals::MultiMapWithErrorMap(int mapIdx,const std::vector<double> &errorMap) {
    std::vector<double> res = maps[mapIdx].GetMapBeforeFun();
    if (res.size() != errorMap.size()) {
        throw std::string("Size mismatch in MultiMapWithErrorMap: res is " + 
                          std::to_string(res.size()) + ", but errorMap is " + std::to_string(errorMap.size()));
    }
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





