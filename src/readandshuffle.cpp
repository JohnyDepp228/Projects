#include "readandshuffle.h"

ReadAndShuffle::ReadAndShuffle()
{

}

ReadAndShuffle::~ReadAndShuffle()
{

}
void ReadAndShuffle::ReadAndShuffleTrain(std::vector<DatasetItem> &vec,std::string path) {

    for (const auto& classEntry : std::filesystem::directory_iterator(path)) {

        if (std::filesystem::is_directory(classEntry.path())) {

            std::string className = classEntry.path().filename().string();
            className.erase(className.find('('),1);
            className.erase(className.find(')'),1);
            for (const auto& fileEntry : std::filesystem::directory_iterator(classEntry.path())) {
                if (std::filesystem::is_regular_file(fileEntry.path())) {
                    vec.push_back({std::atoi(className.c_str()),fileEntry.path().string()});
                }
            }
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);
}