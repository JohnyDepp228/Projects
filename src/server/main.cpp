#include "perceptron.h"
#include <Windows.h>

class Server {
private:
	HANDLE serverPipe;
	DWORD pipeReadLimit = 1024;
	DWORD pipeWriteLimit = 1024;

	DWORD limitOfReadBytes = 1024;

	std::string log;

	bool danger;

public:
	Server() {
		serverPipe = CreateNamedPipeW(
			L"\\\\.\\pipe\\Server_pipe",
			PIPE_ACCESS_DUPLEX,
			PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_ACCEPT_REMOTE_CLIENTS,
			PIPE_UNLIMITED_INSTANCES,
			pipeReadLimit,
			pipeWriteLimit,
			NMPWAIT_WAIT_FOREVER,
			NULL);
		danger = false;


		if (serverPipe == INVALID_HANDLE_VALUE) {
			std::cout << "Can't create pipe\t" << GetLastError() << std::endl;
			return;
		}
	}

	void ReadLog() {
		std::cout << "Waiting for client to write " << std::endl;
		std::cout << "Get signal to read" << std::endl;
		char buffer[1024] = { 0 };
		limitOfReadBytes = 1024;
		DWORD readBytes = 0;
		bool readLog = ReadFile(serverPipe, buffer, limitOfReadBytes, &readBytes, NULL);

		if (!readLog) {
			std::cout << "Error to read from Pipe\t" << GetLastError() << std::endl;
		}
		else {
			this->log = std::string(buffer, readBytes);
		}
	}

	void ClientWait() {
		std::cout << "Waiting for client connection..." << std::endl;
		ConnectNamedPipe(serverPipe, NULL);
		std::cout << "Connected" << std::endl;
	}

	std::string GetLog() const {
		return log;
	}

	void SetDanger(bool danger) {
		this->danger = danger;
	}

	void WriteAnswer() {
		std::string str;
		limitOfReadBytes = 1000;
		DWORD wroteBytes = 0;
		bool writeDanger = WriteFile(serverPipe, &this->danger, sizeof(bool), &wroteBytes, NULL);

		if (!writeDanger) {
			std::cout << "Error to write to Pipe\t" << GetLastError() << std::endl;
		}
		else {
			std::cout << "Sent success" << std::endl;
		}
		FlushFileBuffers(serverPipe);
	}

	~Server() {
		CloseHandle(serverPipe);
	}

};


int main() {

	std::string str;
	Perceptron* p;
	p = new Perceptron(100, 80, 1);

	try {
		p->Start();
	}
	catch (const Errors& e) {
		if (e == Errors::DIVISIONBYZERO) {
			std::cout << "DIVISIONBYZERO" << std::endl;
		}
		else if (e == Errors::FILEREAD) {
			std::cout << "FILEREAD. Error code:\t" << GetLastError() << std::endl;
		}
		else if (e == Errors::FILEREADVEC1D) {
			std::cout << "FILEREADVEC1D. Error code:\t" << GetLastError() << std::endl;
		}
		else if (e == Errors::FILEREADVEC2D) {
			std::cout << "FILEREADVEC2D. Error code:\t" << GetLastError() << std::endl;
		}
		else if (e == Errors::FILEWRITE) {
			std::cout << "FILEWRITE. Error code:\t" << GetLastError() << std::endl;
		}
		else if (e == Errors::FILEWRITEVEC1D) {
			std::cout << "FILEWRITEVEC1D.\t Error code:\t" << GetLastError() << std::endl;
		}
		else if (e == Errors::FILEWRITEVEC2D) {
			std::cout << "FILEWRITEVEC2D.\t Error code:\t" << GetLastError() << std::endl;
		}
	}
	catch (...) {

	}


	Server s;

	s.ClientWait();

	while (true) {
		s.ReadLog();

		std::cout << "Read\t" << s.GetLog() << std::endl;

		s.SetDanger(p->FullProcess(s.GetLog()));

		s.WriteAnswer();
	}

	return 0;
}