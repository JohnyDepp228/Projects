#include "filters.h"

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
                    
                    sum += matrix[idx] * GetElementFromReversed(kx - x,ky - y);

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
    for(auto &f : this->filter){
        f = RandValue(0.5, 0.5);
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
        filter[i] -= LR * errorMat[i]; 
    }
}

std::vector<double> Filters::GetReverseFilter() const {
    return this->revFilter;
}
