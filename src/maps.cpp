#include "maps.h"
#include <algorithm>
#include <iterator>


Maps::Maps()
{

}

Maps::~Maps() {}

std::vector<double> Maps::Pooling(const std::vector<double> &mapsOfSigns) {
    int width;
    int height = width =std::sqrt(mapsOfSigns.size());

    int newHeight = (height - this->poolHeight) / this->stride + 1;
    int newWidth = (width - this->poolWidth) / this->stride + 1;
    int xNewIdx = 0;
    std::vector<double> pooledMap(newHeight * newWidth);
    this->pooledValuesIdx = std::vector<int>(newHeight * newWidth);
    for(int x = 0; x <= height - this->poolHeight; x += this->stride) {
        int yNewIdx = 0;
        for(int y = 0;y <= width - this->poolWidth; y += this->stride) {

            int maxGlobalIdx = x * width + y;
            double maxEl = mapsOfSigns[maxGlobalIdx];

            for(int kx = x; kx < x + this->poolHeight; kx++){
                for(int ky = y; ky < y + this->poolWidth; ky++){
                   int idx = kx * width + ky;
                    if (mapsOfSigns[idx] > maxEl) {
                        maxEl = mapsOfSigns[idx];
                        maxGlobalIdx = idx;
                    }
                }
            }

            int outIdx = xNewIdx * newWidth + yNewIdx;
            this->pooledValuesIdx[outIdx] = maxGlobalIdx;
            pooledMap[outIdx] = maxEl;
            yNewIdx++;

        }
        xNewIdx++;
    }

    return pooledMap;
}

std::vector<double> Maps::ApplyActivateFunction(const std::vector<double> &mapOfSigns) {
    std::vector<double> res = mapOfSigns;
    mapBeforeActivateFunction = mapOfSigns;
    for(auto &el : res){
        if(std::isinf(el) || std::isnan(el)){
            throw std::string("Invalid data SENT to activ function");
            return {};
        }
        el  = GeLu(el);
        if(std::isinf(el) || std::isnan(el)){
            throw std::string("Invalid data AFTER activ function");
            return {};
        }
    }

    return res;
}

void Maps::ApplyDiractivateFunction() {
    for(auto &el : mapBeforeActivateFunction){
        el = DiractiveGeLu(el);
    }

}

std::vector<double> Maps::ReversePooling(const std::vector<double> &mapOfErrors) {
    if(mapOfErrors.size() != pooledValuesIdx.size()){
        throw std::string("Invalid error map size sent");
    }
    else{
        std::vector<double> reversedPoolingMap(this->mapBeforeActivateFunction.size(), 0.0);
        for(int i =0;i < mapOfErrors.size();i++){
            int idx = this->pooledValuesIdx[i];
            reversedPoolingMap[idx] =  mapOfErrors[i];
        }
        return reversedPoolingMap;
    }
}

double Maps::GeLu(double data) {
    return 0.5 * data * (1 + std::tanh(std::sqrt(2 / this->PI) * (data + 0.044715 * std::pow(data, 3))));
}

double Maps::DiractiveGeLu(double data) {
    double y = std::sqrt(2 / this->PI) * (data + 0.044715 * std::pow(data, 3));
	return 0.5 * (1 + tanh(y)) + 0.5 * data * (1 - std::pow(tanh(y), 2)) * std::sqrt(2 / this->PI) 
        * (1 + 0.134145 * std::pow(data, 2));
}

double Maps::GMP() {
    auto maxEl = std::max_element(this->mapOfsigns.begin(),this->mapOfsigns.end());
    this->GMPIdx = std::distance(this->mapOfsigns.begin(),maxEl);
    return *maxEl;
}

std::vector<double> Maps::operator* (const std::vector<double> &reversePoolingMap) const {
    if(this->mapBeforeActivateFunction.size() != reversePoolingMap.size()){
        throw std::string("Can't multi maps with different size");
    }
    else{
        std::vector<double> res(mapBeforeActivateFunction.size());
        for(int i = 0;i < mapBeforeActivateFunction.size();i++){
            res[i] = mapBeforeActivateFunction[i] * reversePoolingMap[i];
        }
        return res;
    }
   
}

void Maps::FindMapForLearning(const std::vector<double> &mapOfErrors) {
    std::vector<double> reversedPoolingMap = ReversePooling(mapOfErrors);
    ApplyDiractivateFunction();
    this->mapOfErrors = (*this) * reversedPoolingMap;
}

void Maps::SetHeight(unsigned int height) { this->mapHeight = height; }

void Maps::SetWidth(unsigned int width) { this->mapWidth = width; }

void Maps::SetMap(const std::vector<double> &mapOfSigns) { this->mapOfsigns = mapOfSigns; }

void Maps::SetStride(int stride) { this->stride = stride; }

void Maps::SetMapBeforeActivate(const std::vector<double> &map) {
    this->mapBeforeActivateFunction = map;
}

unsigned int Maps::GetHeight() const { return this->mapHeight; }

unsigned int Maps::GetWidth() const { return this->mapWidth; }

std::vector<double> Maps::GetMapOfSigns() const { return this->mapOfsigns; }

std::vector<double> Maps::GetMapOfErros() const {
    return this->mapOfErrors;
}


int Maps::GetStride() const { return this->stride; }

std::vector<int> Maps::GetPooledIdx() const { return this->pooledValuesIdx; }

int Maps::GetGMPIdx() const {
    return this->GMPIdx;
}
