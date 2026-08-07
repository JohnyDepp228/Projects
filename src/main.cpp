#include "perceptron.h"
#include <Windows.h>

class Server {
private:
	HANDLE serverPipe;
	DWORD pipeReadLimit = 1024;
	DWORD pipeWriteLimit = 1024;

	DWORD limitOfReadBytes = 1024;
	HANDLE eventToRead;
	HANDLE eventClientRead;

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

		eventToRead = CreateEventW(NULL, FALSE, FALSE, L"ReadServerEvent");

		eventClientRead = CreateEventW(NULL, FALSE, FALSE, L"ReadClientEvent");

		if (serverPipe == INVALID_HANDLE_VALUE) {
			std::cout << "Can't create pipe\t" << GetLastError() << std::endl;
			return;
		}

		if (eventToRead == INVALID_HANDLE_VALUE) {
			std::cout << "Can't create eventToRead\t" << GetLastError() << std::endl;
			return;
		}

		if (eventClientRead == INVALID_HANDLE_VALUE) {
			std::cout << "Can't create eventClientRead\t" << GetLastError() << std::endl;
			return;
		}
	}

	void ReadLog() {
		std::cout << "Waiting for client to write " << std::endl;
		WaitForSingleObject(eventToRead, INFINITE);
		std::cout << "Get signal to read" << std::endl;
		char buffer[1024] = { 0 };
		limitOfReadBytes = 1000;
		DWORD readBytes = 0;
		bool readLog = ReadFile(serverPipe, &buffer, limitOfReadBytes, &readBytes, NULL);

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
		SetEvent(eventClientRead);
	}

	~Server() {
		CloseHandle(serverPipe);
		CloseHandle(eventToRead);
		CloseHandle(eventClientRead);
	}
	
};


int main() {
	
	std::string str;

	Perceptron p(100, 80, 1);

	Server s;

	s.ClientWait();

	s.ReadLog();

	std::cout << "Read\t" << s.GetLog() << std::endl;

	s.SetDanger(p.FullProcess(s.GetLog()));

	s.WriteAnswer();




	return 0;
}