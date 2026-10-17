#include "convolutionallayer.h"
#include "image.h"

ConvolutionalLayer::ConvolutionalLayer()
{

}

ConvolutionalLayer::~ConvolutionalLayer()
{

}
void ConvolutionalLayer::SetChannalsNum(int channalsNum) {
    this->channalNum = channalsNum;
    channals.resize(channalsNum);
}

void ConvolutionalLayer::SetChannalsSize(int channalsSize) {
    this->channalsSize = channalsSize;
}

std::vector<double> ConvolutionalLayer::Forward() {
    std::vector<double> res(1,0.0);
    if(this->channalNum <= 0){
        throw std::string("Invalid channals number");
    }
    else if(this->channalsSize <= 0){
        throw std::string("Invalid channals size");
    }
    else{
        std::vector<double> temp;
        Image im("C:/Users/LordMegatron/Desktop/2.jpg");
        std::vector<double> R = im.GetRedMatrix();
        channals[0].SetFiltersParams(channalsSize, 3, 3);
        channals[0].SetChannalSize(channalsSize);
        temp  = channals[0].ConvResRGB(im.GetRedMatrix(),im.GetGreenMatrix(), im.GetBlueMatrix());
        channalsSize*= 2;
        for(int i =1;i < this->channalNum - 1;i++){
            channals[i].SetFiltersParams(channalsSize, 3, 3);
            
            channals[i].SetChannalSize(channalsSize);
            
            temp = channals[i].ChannalRes(temp);
           
            channalsSize*= 2;
        }
        channals[channalNum -1].SetFiltersParams(channalsSize, 3, 3);
        channals[channalNum -1].SetChannalSize(channalsSize);
        res = channals[channalNum -1].ChannalGMP(temp);
    }

    return res;
}
