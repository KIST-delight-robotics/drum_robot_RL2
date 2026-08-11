#pragma once
 
#include <string>
#include <atomic>
#include <iostream>
#include <string>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <algorithm>
#include <cctype>

#include <cmath>
#include <sstream>
#include <iomanip>
 
#include "common/app_context.hpp"
#include "common/command_queue.hpp"
#include "tcp/command_parser.hpp"
 
class TcpServer {
public:
    TcpServer(AppContext &ctxRef, int port, CommandQueue &commandQueueRef);
    ~TcpServer();
 
    void run();
    void stop();
 
private:
    AppContext &ctx;
    int port_;
    int server_fd_;
    std::atomic<bool> running_;
 
    bool quitting = false;
    CommandQueue &command_queue;

    CommandParser command_parser;
 
    void set_server();
};
 