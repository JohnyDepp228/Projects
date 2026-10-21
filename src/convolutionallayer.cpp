#include "convolutionallayer.h"
#include "channals.h"
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
        std::vector<double> G = im.GetGreenMatrix();
        std::vector<double> B = im.GetBlueMatrix();

        std::vector<double> grayInput(R.size());
        for(size_t i = 0; i < R.size(); i++) {
            grayInput[i] = 0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i];
        }

        channals[0].SetInput(grayInput); 
        channals[0].SetFiltersParams(channalsSize, 3, 3);
        channals[0].SetChannalSize(channalsSize);
        temp = channals[0].ConvResGray(grayInput); 
        channalsSize*= 2;
        for(int i =1;i < this->channalNum - 1;i++){
            std::cout << "Channal idx: " << i << std::endl;
            channals[i].SetFiltersParams(channalsSize, 3, 3);
            
            channals[i].SetChannalSize(channalsSize);

            channals[i].SetInput(temp);
            
            temp = channals[i].ChannalRes(temp);
           
            channalsSize*= 2;
        }
        channals[channalNum -1].SetFiltersParams(channalsSize, 3, 3);
        channals[channalNum -1].SetChannalSize(channalsSize);
        channals[channalNum -1].SetInput(temp);
        res = channals[channalNum -1].ChannalGMP(temp);
    }

    return res;
}

void ConvolutionalLayer::Backpropageation(const std::vector<double> &classifierError) {
    try{
       channals[channalNum - 1].ChannalLearnGMP(classifierError);
        std::cout << "\nSuccess in GMP learn\n";
        
        if (channalNum > 1) {
            SetPrevLayerErrorMap(channalNum - 1, channalNum - 2);
        }

      
        for(int i = channalNum - 2; i >= 0; i--) {
            std::cout << "Current layer:  " << i << std::endl;
            channals[i].ChannalLearn();
            
            
            if (i > 0) {
                SetPrevLayerErrorMap(i, i - 1);
            }
        }
    }
    catch(const std::string &msg){
        std::cout << "Error in backpropagation " + msg;
    }
}

void ConvolutionalLayer::SetPrevLayerErrorMap(int currentChannelIdx,int prevChannelIdx) {
    int currentSize = channals[currentChannelIdx].GetChannelSize();
    int prevSizez = channals[prevChannelIdx].GetChannelSize();
    if(currentChannelIdx > currentSize || currentChannelIdx < 0 ){
        throw std::string("Current channel idx is invalid");
    }
    else if(prevChannelIdx > prevSizez || prevChannelIdx < 0 ){
        throw std::string("Prev channel idx is invalid");
    }
    else{
        int size = channals[prevChannelIdx].GetChannelSize();
        for(int i =0;i < prevSizez;i++){
            if (i >= currentSize) {
            std::cout << "Trying take map" << i 
                      << " from channel " << currentChannelIdx 
                      << "but only " << currentSize << " maps "<< std::endl;
            break; 
            }

            std::vector<double> errorMap = channals[currentChannelIdx].GetPrevLayerError(i);
            if(errorMap.size() == 0) {
                std::cout << "Error in channel: " << currentChannelIdx 
                  << " empty errormap " << i << std::endl;
                }
            channals[prevChannelIdx].SetErrorMap(i, errorMap);
        }
    }
       

}
