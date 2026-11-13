#include <iostream> 
#include "filters.h"


int main(){

    Filters f;
    std::vector<double> bigMat(16,2.0);
    std::vector<double> smallMat(4,4.0);
    try{
        std::vector<double> res = f.Slide(bigMat, smallMat);
    }
    catch(std::string msg){
        std::cout << msg;
    }
    

    return 0;
}