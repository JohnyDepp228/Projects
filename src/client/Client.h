#ifndef CLIENT_H
#define CLIENT_H

#include <iostream>
#include <Windows.h>
#include <vector>
#include <random>
#include <string>


class Client {

private:
    std::vector<std::string> SQL;
    std::vector<std::string> XSS;
    std::vector<std::string> Win;
    std::vector<std::string> methods;
    std::vector<std::string> pages;
    std::vector<std::string> protocol;
    std::vector<std::string> agent;

    std::string request;

    HANDLE requstToServer;


public:
    Client() {
        Client::InitLogsDB();
        requstToServer = CreateFileW(L"\\\\.\\pipe\\Server_pipe",
            GENERIC_READ | GENERIC_WRITE,
            0,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);



        if (requstToServer == INVALID_HANDLE_VALUE) {
            std::cout << "Invalid requstToServer\t" << GetLastError() << std::endl;
        }

    }

    int Index(int leftBoard, int rightBoard);

    void InitLogsDB();

    std::string CreateRequest();

    bool DangerLog() {
        return (int)Index(0, 1);
    }

    void SendRequest();

    bool ReadServerAnswer();

    ~Client() {
        CloseHandle(requstToServer);
    }
};

#endif 
