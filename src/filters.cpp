#include "filters.h"
#include <cmath>
#include <string>

Filters::Filters()
{
    
}

Filters::~Filters()
{
    
}

 void Filters::SetSize(int height,int width){
    this->width = width;
    this->height = height;
    this->filter.resize(this->width * this->height,0.0);
 }


double Filters::GetElement(int x,int y) const
    {
        return this->filter[(x * this->width) + y]; 
    }
double Filters::GetElementFromReversed(int x,int y) const 
    {
        return this->revFilter[(x * this->width) + y];
    }
std::vector<double> Filters::Slide(const std::vector<double> &matrix,int stride) {
    int height = std::sqrt(matrix.size());
    int width = std::sqrt(matrix.size());
    int widthStart = width;
    int heightStart = height;
    height -= this->height;
    width -= this->width;
    int newWidth = ((widthStart - this->width) / stride) + 1;
    int newHeight  =((heightStart - this->height) / stride) + 1;
    std::vector<double> temp(newWidth * newHeight,0.0);
    int xTemp = 0;
    for(int x = 0;x <= height; x += stride){
        int yTemp = 0;
        for(int y =0;y <= width; y += stride){
            double sum = 0.0;
            for(int kx  = x;kx <  x + this->height;kx++){
                for(int ky = y;ky < y + this->width;ky++){
                    
                    int idx = kx * widthStart  + ky;
                    
                    sum += matrix[idx] * GetElement(kx - x,ky - y);

                }
            }
            sum += bias;
            int tempIdx = xTemp * newWidth + yTemp;
            temp[tempIdx] = sum;
            yTemp++;
        }
        xTemp++;
    }

    return temp;
}

std::vector<double> Filters::SlideWithReversed(const std::vector<double> &matrix,int stride) {
    int height = std::sqrt(matrix.size());
    int width = std::sqrt(matrix.size());
    int widthStart = width;
    int heightStart = height;
    height -= this->height;
    width -= this->width;
    int newWidth = ((widthStart + this->width) / stride) - 1;
    int newHeight  =((heightStart + this->height) / stride) - 1;
    std::vector<double> temp(newWidth * newHeight,0.0);
    int xTemp = 0;
    for(int x = 0;x <= height; x += stride){
        int yTemp = 0;
        for(int y =0;y <= width; y += stride){
            double sum = 0.0;
            for(int kx  = x;kx <  x + this->height;kx++){
                for(int ky = y;ky < y + this->width;ky++){
                    
                    int idx = kx * widthStart  + ky;
                    if(this->revFilter.size() <= 0){
                        throw std::string("Reversed filter is empty");
                    }
                    sum += matrix[idx] * GetElementFromReversed(kx - x,ky - y);

                }
            }
           
            int tempIdx = xTemp * newWidth + yTemp;
            int filterSize = this->width  * this->height;
            temp[tempIdx] = sum / filterSize;
            yTemp++;
        }
        xTemp++;
    }

    return temp;
}

std::vector<double> Filters::SlideRGB(const std::vector<double> &R,const std::vector<double> &G,const std::vector<double> &B,int stride) {
    int height = std::sqrt(R.size());
    int width = std::sqrt(R.size());
    int widthStart = width;
    int heightStart = height;
    height -= this->height;
    width -= this->width;
    int newWidth = ((widthStart - this->width) / stride) + 1;
    int newHeight  =((heightStart - this->height) / stride) + 1;
    std::vector<double> temp(newWidth * newHeight,0.0);
    int xTemp = 0;
    for(int x = 0;x <= height; x += stride){
        int yTemp = 0;
        for(int y =0;y <= width; y += stride){
            double sum = 0.0;
            for(int kx  = x;kx <  x + this->height;kx++){
                for(int ky = y;ky < y + this->width;ky++){
                    
                    int idx = kx * widthStart + ky;
                    
                    sum += R[idx] * GetElement(kx - x,ky - y);
                    sum += G[idx] * GetElement(kx - x,ky - y);
                    sum += B[idx] * GetElement(kx - x,ky - y);

                }
            }
            sum += this->bias;
            int tempIdx = xTemp * newWidth + yTemp;
            temp[tempIdx] = sum;
            yTemp++;
        }
        xTemp++;
    }

    return temp;
}



void Filters::FillFilter() {
    double bound = std::sqrt(2.0 / (width * height)); 
    for(auto &f : this->filter){
        f = RandValue(-bound, bound);
    }
}

double Filters::RandValue(double leftBorder,double rightBorder) {
    std::random_device rd;
    std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(leftBorder, rightBorder);
	return dis(gen);
}

void Filters::LearnFilters(const std::vector<double> &errorMat) {
    revFilter = filter;
    std::reverse(revFilter.begin(),revFilter.end());

    
    int size = width * height;
    for(int i =0;i < size;i++){
        if (!std::isfinite(errorMat[i])) {
            throw std::string("Nan/ inf in learning filters in errorMat at " + std::to_string(i) + "\n"); 
        }
        filter[i] -= LR * (errorMat[i] / size); 
    }

    for(int i =0;i < filter.size();i++){
        if(!std::isfinite(filter[i])){
           throw std::string("Invalid filter after learn at " + std::to_string(i) + "\t");
        }
    }
}

std::vector<double> Filters::GetReverseFilter() const {
    return this->revFilter;
}
