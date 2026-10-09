#include "mapofsigns.h"

MapOfSigns::MapOfSigns()
{

}

MapOfSigns::~MapOfSigns() 
{
   
}
std::vector<double> MapOfSigns::GetMap() const { return this->mapOfSigns; }
std::vector<double> MapOfSigns::GetMapBeforeFun() const { return this->mapOfSignsBeforeActiveFunction; }
std::vector<int> MapOfSigns::PoolingMaxIdx() const {
    return this->poolingMaxNumIdx;
}
void MapOfSigns::SetMap(const std::vector<double> &map) {
    this->mapOfSigns = map;
}
void MapOfSigns::SetMapBeforeFun(const std::vector<double> &mapBeforeFun) {
    this->mapOfSignsBeforeActiveFunction = mapBeforeFun;
}

std::vector<double> MapOfSigns::Pooling(const std::vector<double> &matrix) {
    int heightStart = std::sqrt(matrix.size());
    int widthStart = std::sqrt(matrix.size());
    int height = heightStart - this->poolingHeight;
    int width = widthStart - this->poolingWidth;
    int newWidth = ((widthStart - this->poolingWidth) / poolingStride) + 1;
    int newHeight = ((heightStart - this->poolingHeight) / poolingStride) + 1;
    std::vector<double> temp(newWidth  *newHeight);
    this->poolingMaxNumIdx.resize(newWidth  * newHeight);
    int xTemp = 0;
    for(int x = 0;x <= height; x += poolingStride){
        int yTemp = 0;
        for(int y =0;y <= width; y += poolingStride){
            double sum = 0.0;
            std::vector<double> poolMat(this->poolingHeight *this->poolingWidth );
            for(int kx  = x;kx <  x + this->poolingHeight;kx++){
                for(int ky = y;ky < y + this->poolingWidth;ky++){
                    int idx = kx * widthStart  + ky;
                    int localPoolIdx = (kx - x) * this->poolingWidth + (ky - y);
                    
                    poolMat[localPoolIdx] = matrix[idx];   
                }
            }
            auto maxElem = std::max_element(poolMat.begin(),poolMat.end());
           
            int maxIdxInPool = std::distance(poolMat.begin(), maxElem);
            int tempIdx = xTemp * newWidth + yTemp;
            temp[tempIdx] = *maxElem;
            poolingMaxNumIdx[tempIdx] =  maxIdxInPool;
            yTemp++;
        }
        xTemp++;
    }

    return temp;
}

std::vector<double> MapOfSigns::ReversePooling(const std::vector<double> &errorMatrix,int stride) {

  return {};
}
