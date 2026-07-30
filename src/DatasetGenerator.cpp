#include "DatasetGenerator.h"


double Dataset::Index(double leftBoard, double rightBoard) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
	return dis(gen);
}

void Dataset::InitLogsDB() {
    std::vector<std::string> methods = { "GET", "POST", "PUT", "DELETE" };
    std::vector<std::string> pages = { "/index.html", "/login.php", "/profile", "/shop/items", "/assets/style.css", "/js/main.js", "/contacts" };
    std::vector<std::string> protocol = { "HTTP/1.1", "HTTP/2.0" };
    std::vector<std::string> agent = { "Mozilla/5.0", "Chrome/120.0", "Safari/13.1", "Curl/7.68.0" };


    std::vector<std::string> SQL = { "' OR '1'='1'", "admin' --", "' UNION SELECT NULL, username, password FROM users --",
    "1; DROP TABLE users; --", "' OR 1=1 --'", "1' AND 1=2 UNION SELECT", "OR 1=1" };
    std::vector<std::string> XSS = { "<script>alert('xss')</script>", "javascript:alert(1)", "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>", "element.innerHTML = <script>" };
    std::vector<std::string> Win = { "../../../../etc/passwd", "..\\..\\..\\windows\\win.ini", "/etc/shadow",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "../boot.ini" };
    for (int i = 0; i < Dataset::testListSize; i++) {
        logs.push_back(CreateLog(methods, pages, protocol, agent, SQL, XSS, Win,i));
    }
}

std::string Dataset::CreateLog(const std::vector<std::string>& methods, const std::vector<std::string>& pages, const std::vector<std::string>& protocol,
    const std::vector<std::string>& agent,
    const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win,const int &index)
{
    std::string res = methods[(int)Index(0,methods.size())] + " " +  pages[(int)Index(0, pages.size())] + " " + protocol[(int)Index(0, protocol.size())]
        + " " + agent[Index(0, agent.size())]+ " " ;
    if ((int)Index(0, 10) % 2 == 0) {
        Dataset::dangerLevel[index] = 1;
        switch ((int)Index(1, 3)) {
        case 1: res += SQL[(int)Index(0, SQL.size())]; break;
        case 2: res += XSS[(int)Index(0, XSS.size())]; break;
        case 3: res += Win[(int)Index(0, Win.size())]; break;
        }
    }
    else {
        Dataset::dangerLevel[index] = 0;
    }
    return res;

}

bool Dataset::DangerLog() {
    return (int)Index(0, 1);
}