#include "config.h"
#include <cerrno>
#include <sstream>

Config::Config()
{

}

Config::~Config()
{

}
void Config::SetPath(const std::string &path) {
    this->path = path;
}

bool Config::SaveData(std::string dataType, int data) {
    bool res = false;
    std::fstream out(this->path,std::ios::out | std::ios::trunc);
    if(out.is_open()){
        out << dataType << ": " << data << "\n";
        res = true;
    }
    else{
        out.close();
        throw std::string("Can't save data to file. Error code: " + std::to_string(errno));
    }
    out.close();
    return res;
}

bool Config::SaveData(std::string dataType,const std::vector<double> &data) {
    bool res = false;
    std::fstream out(this->path,std::ios::out | std::ios::app);
    if(out.is_open()){
        out << dataType << ": ";
        for(const auto &n :data){
            out << n << ' ';
        }
        out << "\n";
        res = true;
    }
    else{
        out.close();
        throw std::string("Can't save data to file. Error code: " + std::to_string(errno));
    }
    out.close();
    return res;
}

void Config::ReadData(std::string dataType, int &data) {
    std::fstream in(this->path,std::ios::in);
    if(in.is_open()){
        std::string temp;
        std::getline(in, temp);
        if(temp.find(dataType) != std::string::npos){
                size_t pos = temp.find(':');
                temp = temp.substr(pos + 1);
                data = std::stoi(temp);
        }
    }
    else{
        in.close();
        throw std::string("Can't save data to file. Error code: " + std::to_string(errno));
    }
    in.close();
}
void Config::ReadData(std::string dataType, std::vector<double> &data) {
    std::fstream in(this->path,std::ios::in);
    if(in.is_open()){
        std::string temp;
        while(std::getline(in, temp)){
            if(temp.find(dataType) != std::string::npos){
                size_t pos = temp.find(':');
                temp = temp.substr(pos + 1);
                std::istringstream iss(temp);
                double weight;
                while(iss >> weight){
                    data.push_back(weight);
                }
                break;
            }
        }
    }
    else{
        in.close();
        throw std::string("Can't save data to file. Error code: " + std::to_string(errno));
    }
    in.close();
}

