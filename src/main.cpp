#include <iostream> 
#include "channel.h"
#include "maps.h"


int main() {
   
try{
    std::vector<Channel> channel;
    channel.resize(2);
    std::vector<Maps> map(2);
    std::vector<Maps> mapRes;
    std::vector<double> mat(64,1.0);
    std::vector<double> mat2(64,2.0);

    map[0].SetMap(mat);
    map[0].SetHeight(8);
    map[0].SetWidth(8);
    map[1].SetMap(mat2);
    map[1].SetHeight(8);
    map[1].SetWidth(8);


    channel[0].SetSize(2);
    mapRes = channel[0].Forward(map);
    std::cout << "Map size after forward 0" << std::endl;
    for (int i = 0; i < mapRes.size(); i++) {
    std::cout << "Map " << i
              << " size: " << mapRes[i].GetMapOfSigns().size()
              << '\n';
    }
    channel[1].SetSize(3);
    mapRes = channel[1].Forward(mapRes);
    std::cout << "Map size after forward 1" << std::endl;
    for (int i = 0; i < mapRes.size(); i++) {
    std::cout << "Map " << i
              << " size: " << mapRes[i].GetMapOfSigns().size()
              << '\n';
    }


    }
catch(std::string &msg){
        std::cout << msg;
    }
catch(std::exception& e){
    std::cout << e.what();
}
catch(...){
        std::cout << "nn";
    }

    return 0;
}
