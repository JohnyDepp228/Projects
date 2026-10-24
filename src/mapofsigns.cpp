#include "mapofsigns.h"


MapOfSigns::MapOfSigns()
{

}

MapOfSigns::~MapOfSigns() 
{
   
}
std::vector<double> MapOfSigns::GetMap() const { return this->mapOfSigns; }
std::vector<double> MapOfSigns::GetMapBeforeFun() const { return this->mapOfSignsBeforeActiveFunction; }
std::vector<int> MapOfSigns::GetPoolingMaxIdx() const { return this->poolingMaxNumIdx; }
std::vector<double> MapOfSigns::GetErrorMap() const {
    return this->errorMat;
}
std::vector<double> MapOfSigns::GetPrevLayerError() const {
    return this->PrevLayerError;
}
void MapOfSigns::SetMap(const std::vector<double> &map) { this->mapOfSigns = map;this->width = this->height = std::sqrt(map.size());}
void MapOfSigns::SetMapBeforeFun(const std::vector<double> &mapBeforeFun) { this->mapOfSignsBeforeActiveFunction = mapBeforeFun; }

void MapOfSigns::SetErrorMap(const std::vector<double> &errorMap) {
    this->errorMat = errorMap;
}

void MapOfSigns::SetPrevLayerError(const std::vector<double> &preLayerError) {
    this->PrevLayerError = preLayerError;
}

void MapOfSigns::SetOriginalInput(const std::vector<double> &input) {
    this->originalInput = input;
}

std::vector<double> MapOfSigns::Pooling(const std::vector<double> &matrix) {
    int heightStart = std::sqrt(matrix.size());
    int widthStart = std::sqrt(matrix.size());
    int poolHeightLimit = heightStart - this->poolingHeight;
    int poolWidthLimit = widthStart - this->poolingWidth;
    int widthAfterPool = ((widthStart - this->poolingWidth) / poolingStride) + 1;
    int heightAfterPool = ((heightStart - this->poolingHeight) / poolingStride) + 1;
    std::vector<double> res(widthAfterPool  *heightAfterPool);
    this->poolingMaxNumIdx.resize(widthAfterPool  * heightAfterPool);
    int resX = 0;
    for(int x = 0;x <= poolHeightLimit; x += poolingStride){
        int resY = 0;
        for(int y =0;y <= poolWidthLimit; y += poolingStride){
            double sum = 0.0;
            std::vector<double> poolMat(this->poolingHeight *this->poolingWidth );
            std::vector<int> globalIdx;
            for(int kx  = x;kx <  x + this->poolingHeight;kx++){
                for(int ky = y;ky < y + this->poolingWidth;ky++){
                    int idx = kx * widthStart  + ky;
                    int localPoolIdx = (kx - x) * this->poolingWidth + (ky - y);
                    
                    poolMat[localPoolIdx] = matrix[idx];   
                    globalIdx.push_back(idx);
                }
            }
            auto maxElem = std::max_element(poolMat.begin(),poolMat.end());
           
            int maxIdxInPool = std::distance(poolMat.begin(), maxElem);
            int tempIdx = resX * widthAfterPool + resY;
            res[tempIdx] = *maxElem;

            poolingMaxNumIdx[tempIdx] =  globalIdx[maxIdxInPool];
            resY++;
        }
        resX++;
    }

    return res;
}

std::vector<double> MapOfSigns::ReversePooling(int targetSize) {
    std::vector<double> res(targetSize,0.0);
    if(errorMat.size() <= 0){
        throw std::string("prev layer size <= 0");
    }
    //int size = std::sqrt(PrevLayerError.size()) * this->poolingStride; 

    int safeLimit = std::min(poolingMaxNumIdx.size(), errorMat.size());

    for(int i = 0; i < safeLimit; i++){
        int idx = poolingMaxNumIdx[i];
        
        if (idx < res.size()) {
            res[idx] = errorMat[i];
        }
    }
    return res;
}

void MapOfSigns::ApplyActivFun( std::vector<double> &matrix) {
    for(int i =0;i < matrix.size();i++){
        matrix[i] = GeLu(matrix[i]);
    }
}

void MapOfSigns::ApplyDiractiveActivFun(std::vector<double> &errorMat) {
    if(errorMat.size() != mapOfSignsBeforeActiveFunction.size()){
        throw std::string("Size of errorMat != size of mamapOfSignsBeforeActiveFunction");
    }
    for(int i = 0; i< errorMat.size(); i++){
        errorMat[i] *= DiractiveGeLu(mapOfSignsBeforeActiveFunction[i]);
    }
}


double MapOfSigns::GeLu(double data){
    return 0.5 * data * (1 + std::tanh(std::sqrt(2 / this->PI) * (data + 0.044715 * std::pow(data, 3))));

}

double MapOfSigns::DiractiveGeLu(double data) {
    double y = std::sqrt(2 / this->PI) * (data + 0.044715 * std::pow(data, 3));
	return 0.5 * (1 + tanh(y)) + 0.5 * data * (1 - std::pow(tanh(y), 2)) * std::sqrt(2 / this->PI) 
        * (1 + 0.134145 * std::pow(data, 2));

}

double MapOfSigns::GMP() {
    auto max = std::max_element(this->mapOfSigns.begin(), this->mapOfSigns.end());
    this->GMPIdx = std::distance(this->mapOfSigns.begin(), max);
    return *max;
}

int MapOfSigns::GetGMPIDx() const {
    return this->GMPIdx;
}

std::vector<double> MapOfSigns::MapsSlide(const std::vector<double> &errorMap) {
    std::cout << "DEBUG MapsSlide -> originalInput size: " << originalInput.size() 
              << " | errorMap size: " << errorMap.size() << std::endl;
    if(originalInput.size() == 0){
        throw std::string("Empty mapOfSigns before active function");
    }


    for (size_t i = 0; i < errorMap.size(); ++i) {
        if (std::isnan(errorMap[i])) {
            std::cout << "In MapsSlide sent errorMap, already with NaN at " << i << std::endl;
            throw std::string("NaN detected in input errorMap");
        }
    }
    int smallerMatHeight = std::sqrt(errorMap.size());
    int smallerMapWidth = std::sqrt(errorMap.size());
    int biggerMatWidth = std::sqrt(originalInput.size());
    int biggerMatHeight = std::sqrt(originalInput.size());

    int newWidth = ((biggerMatWidth - smallerMapWidth) / 1) + 1;
    int newHeight  =((biggerMatHeight - smallerMatHeight) / 1) + 1;
    std::vector<double> temp(newWidth * newHeight,0.0);
    for(int x = 0;x < newHeight; x += 1){
       
        for(int y =0;y < newWidth; y += 1){
            double sum = 0.0;
            for(int kx  = 0;kx <  smallerMatHeight;kx++){
                for(int ky = 0;ky < smallerMapWidth;ky++){
                    
                    int errorIdx = kx * smallerMapWidth  + ky;

                    int idx = (x + kx) * biggerMatWidth + (y + ky);
                    
                    sum += originalInput[idx] * errorMap[errorIdx];

                }
            }
            int tempIdx = x * newWidth + y;
            temp[tempIdx] = sum;

              if (std::isnan(sum)) {
                std::cout << "!!! Catch NaN in MapsSlide x=" << x << ", y=" << y << std::endl;
                throw std::string("NaN generated inside MapsSlide loop");
            }
        }
    }

    

    return temp;
}


