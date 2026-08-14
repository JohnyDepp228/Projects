#include "Client.h"

int Client::Index(int leftBoard, int rightBoard) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(leftBoard, rightBoard);
    return dis(gen);
}

void Client::InitLogsDB() {
    this->methods = { "GET", "POST", "PUT", "DELETE" };
    this->pages = { "/index.html", "/login.php", "/profile", "/shop/items", "/assets/style.css", "/js/main.js", "/contacts" };
    this->protocol = { "HTTP/1.1", "HTTP/2.0" };
    this->agent = { "Mozilla/5.0", "Chrome/120.0", "Safari/13.1", "Curl/7.68.0" };


    this->SQL = { "' OR '1'='1'", "admin' --", "' UNION SELECT NULL, username, password FROM users --",
    "1; DROP TABLE users; --", "' OR 1=1 --'", "1' AND 1=2 UNION SELECT", "OR 1=1" };
    this->XSS = { "<script>alert('xss')</script>", "javascript:alert(1)", "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>", "element.innerHTML = <script>" };
    this->Win = { "../../../../etc/passwd", "..\\..\\..\\windows\\win.ini", "/etc/shadow",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "../boot.ini" };
}

std::string Client::CreateRequest()
{
    std::string res = this->methods[(int)Index(0, this->methods.size() - 1)] + " " + this->pages[(int)Index(0, this->pages.size() - 1)] + " " +
        this->protocol[(int)Index(0, this->protocol.size() - 1)]
        + " " + this->agent[Index(0, this->agent.size() - 1)] + " ";
    if (DangerLog() == 1) {
        int attackType = Index(1, 3);
        switch (attackType) {
        case 1: res += this->SQL[Index(0, this->SQL.size() - 1)]; break;
        case 2: res += this->XSS[Index(0, this->XSS.size() - 1)]; break;
        case 3: res += this->Win[Index(0, this->Win.size() - 1)]; break;
        }
    }
    return res;
}

void Client::SendRequest() {
    DWORD wroteBytes = 0;
    request = CreateRequest();
    std::cout << "Sending\t" << request << "\tSizeof " << request.size() << std::endl;

    bool writeToPipe = WriteFile(requstToServer, (LPVOID)request.data(), sizeof(char) * request.size(), &wroteBytes, NULL);
    if (!writeToPipe) {
        std::cout << "Can't send to pipe\t" << GetLastError() << std::endl;
        return;
    }
    else {
        std::cout << "Sent success" << std::endl;
    }
    FlushFileBuffers(requstToServer);
}

bool Client::ReadServerAnswer() {
    std::cout << "Waiting for serve to answer " << std::endl;
    DWORD readBytes = 0;
    bool danger = false;
    bool readFromPipe = ReadFile(requstToServer, &danger, sizeof(bool), &readBytes, NULL);
    if (!readFromPipe) {
        std::cout << "Can't read from pipe\t" << GetLastError() << std::endl;
        exit(1);
    }
    if (readBytes < sizeof(bool)) {
        std::cout << "Read less data from server" << std::endl;
        exit(1);
    }


    return danger;
}

