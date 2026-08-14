#include "Client.h"

int main()
{
    Client c;
    while (true) {
        c.CreateRequest();
        c.SendRequest();
        if (c.ReadServerAnswer()) {
            std::cout << "Danger!!!" << std::endl;
            Sleep(1000);
        }
    }

    return 0;
}