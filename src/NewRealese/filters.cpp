#include "filters.h"

Filters::Filters()
{
    SetSize(3,3);
    //std::cout << GetElement(0,1);
}

Filters::~Filters()
{
    
}

 void Filters::SetSize(int height,int width){
    this->width = width;
    this->height = height;
    this->filter.resize(this->width * this->height,0.0);
 }

//  double& Filters::operator()(int x,int y)
//     {
//         return this->filter[x * this->width + y];
//     }

double Filters::GetElement(int x,int y) const
    {
        return this->filter[(x * this->width) + y]; 
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
                    int filterIdx = kx + this->width + ky;
                    int idx = kx * widthStart  + ky;
                    
                    sum += matrix[idx] * GetElement(kx - x,ky - y);
                   // std::cout << matrix[idx] << "\t";

                }
            }
            int tempIdx = xTemp * newWidth + yTemp;
            temp[tempIdx] = sum;
            yTemp++;
        }
        xTemp++;
    }

    return temp;
}
