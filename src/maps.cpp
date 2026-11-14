#include "maps.h"

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

void Maps::SetHeight(unsigned int height) { this->mapHeight = height; }

void Maps::SetWidth(unsigned int width) { this->mapWidth = width; }

void Maps::SetMap(const std::vector<double> &mapOfSigns) { this->mapOfsigns = mapOfSigns; }

void Maps::SetStride(int stride) { this->stride = stride; }

unsigned int Maps::GetHeight() const { return this->mapHeight; }

unsigned int Maps::GetWidth() const { return this->mapWidth; }

std::vector<double> Maps::GetMapOfSigns() const { return this->mapOfsigns; }

int Maps::GetStride() const { return this->stride; }

std::vector<int> Maps::GetPooledIdx() const { return this->pooledValuesIdx; }
