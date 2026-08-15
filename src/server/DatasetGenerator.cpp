#include "DatasetGenerator.h"


int Dataset::Index(int leftBoard, int rightBoard) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(leftBoard, rightBoard);
    return dis(gen);
}

void Dataset::InitLogsDB() {
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


    for (int i = 0; i < Dataset::testListSize; i++) {
        int targetDanger = DangerLog();
        if (targetDanger == 1) {
            Dataset::dangerLevel[i] = 0.9;
        }
        else {
            Dataset::dangerLevel[i] = 0.1;
        }
        logs.push_back(CreateLog(methods, pages, protocol, agent, SQL, XSS, Win, targetDanger));
    }
}

std::string Dataset::CreateLog(const std::vector<std::string>& methods, const std::vector<std::string>& pages, const std::vector<std::string>& protocol,
    const std::vector<std::string>& agent,
    const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win, const int& danger)
{
    std::string res = methods[(int)Index(0, methods.size() - 1)] + " " + pages[(int)Index(0, pages.size() - 1)] + " " + protocol[(int)Index(0, protocol.size() - 1)]
        + " " + agent[Index(0, agent.size() - 1)] + " ";
    if (danger == 1) {
        switch (Index(1, 3)) {
        case 1: res += SQL[Index(0, SQL.size() - 1)]; break;
        case 2: res += XSS[Index(0, XSS.size() - 1)]; break;
        case 3: res += Win[Index(0, Win.size() - 1)]; break;
        }
    }
    return res;

}

bool Dataset::DangerLog() {
    return Index(0, 1);
}