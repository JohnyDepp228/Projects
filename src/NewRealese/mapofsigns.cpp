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
void MapOfSigns::SetMap(const std::vector<double> &map) { this->mapOfSigns = map; }
void MapOfSigns::SetMapBeforeFun(const std::vector<double> &mapBeforeFun) { this->mapOfSignsBeforeActiveFunction = mapBeforeFun; }

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

std::vector<double> MapOfSigns::ReversePooling(const std::vector<double> &errorMatrix) {
    std::vector<double> res;
    int size = std::sqrt(errorMatrix.size()) * this->poolingStride; 

    res.resize(size * size);

    for(int i =0;i < poolingMaxNumIdx.size();i++){
        int idx = poolingMaxNumIdx[i];
        res[idx] = errorMatrix[i];
    }
    return res;
}

void MapOfSigns::ApplyActivFun( std::vector<double> &matrix) {
    for(int i =0;i < matrix.size();i++){
        matrix[i] = GeLu(matrix[i]);
    }
}

void MapOfSigns::ApplyDiractiveActivFun() {
    for(auto &data: mapOfSignsBeforeActiveFunction){
        data = DiractiveGeLu(data);
    }
}


double MapOfSigns::GeLu(double data){
    return 0.5 * data * (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * (data + 0.044715 * std::pow(data, 3))));

}

double MapOfSigns::DiractiveGeLu(double data) {
    double y = std::sqrt(2 / std::numbers::pi) * (data + 0.044715 * std::pow(data, 3));
	return 0.5 * (1 + tanh(y)) + 0.5 * data * (1 - std::pow(tanh(y), 2)) * std::sqrt(2 / std::numbers::pi) 
        * (1 + 0.134145 * std::pow(data, 2));

}
