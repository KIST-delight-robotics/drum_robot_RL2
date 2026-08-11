#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cstring>
#include <fstream>
#include <cstdlib>

#include <fcntl.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/ioctl.h>

#include <net/if.h>
#include <arpa/inet.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#include "nlohmann/json.hpp"

#include "util/hostname.hpp"

class CanInterface {
public:
    CanInterface();
    ~CanInterface();

    void resetCanPorts();                                           // USB 허브 포트 전원을 껐다 켜 CAN(PCAN-USB) 하드웨어 리셋
    void initialize();                                              // CAN 포트 초기화 : 사용가능 포트 확인 및 소켓 할당

    bool sendFrame(const std::string &ifname, const can_frame &frame);
    bool sendFrame(int socket, const can_frame &frame);
 
    bool receiveFrame(const std::string &ifname, can_frame &frame);
    bool receiveFrame(int socket, can_frame &frame);

    bool sendandReceiveFrame(const std::string &ifname, can_frame &frame);
    bool sendandReceiveFrame(int socket, can_frame &frame);

    void setSocketNonBlock();
    void setSocketBlock();

    std::map<std::string, int> getSocket();

    void setSocketsTimeout(int sec, int usec);
    void clearReadBuffers();

private:
    static const int ERR_SOCKET_CREATE_FAILURE = -1;
    static const int ERR_SOCKET_CONFIGURE_FAILURE = -2;

    std::map<std::string, int> sockets;                             // 포트 이름 -> 소켓 번호 매핑

    std::vector<std::string> listAndActivateAvailableCANPorts();    // Down 상태인 CAN 포트 활성화 및 활성화 포트 저장
    bool getCanPortStatus(const char *port);                        // CAN 포트 활성화 상태 확인
    void activateCanPort(const char *port);                         // CAN 포트 활성화
    int createSocket(const std::string& ifname);                    // CAN 소켓 생성 및 바인딩

    int setSocketTimeout(int socket, int sec, int usec);
    void clearCanBuffer(int canSocket);
};