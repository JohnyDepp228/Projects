#include <iostream> 
#include "filters.h"
#include "maps.h"


int main() {
    Filters f;
    f.SetStride(1);
    f.SetHeight(3);
    f.SetWidth(3);
    f.FillFilter();
    std::vector<double> errorMap = { 1.0, 1.0, 1.0, 1.0 };
    std::vector<double> prevError;
    try{
        prevError = f.FindMapForPrevLayer(errorMap);
        std::cout << prevError.size();
    }
    catch(std::string *msg){
        std::cout << msg;
    }

    std::vector<double> expected = {
        1.0,  3.0, 2.0,
        4.0, 10.0, 6.0,
        3.0,  7.0, 4.0
    };
}
