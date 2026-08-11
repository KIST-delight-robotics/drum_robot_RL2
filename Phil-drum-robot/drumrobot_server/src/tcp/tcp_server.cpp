#include "tcp/tcp_server.hpp"

TcpServer::TcpServer(AppContext &ctxRef, int port, CommandQueue &commandQueueRef)
    : ctx(ctxRef), port_(port), server_fd_(-1), running_(false), command_queue(commandQueueRef) {}

TcpServer::~TcpServer() {
    
}

void TcpServer::run() {
    set_server();

    std::cout << "[TcpServer] Listening on port " << port_ << "\n";
    
    while (ctx.running.load()) {
        if (quitting) {
            usleep(100);    // 종료 대기
            continue;
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd_, (sockaddr*)&client_addr, &client_len);   // client 연결될때까지 블로킹
        if (client_fd < 0) {
            if (ctx.running.load()) {
                std::cerr << "[TcpServer] accept() failed\n";
            }
            break;
        }

        std::cout << "[TcpServer] Client connected: "
                  << inet_ntoa(client_addr.sin_addr) << "\n";

        // handle client
        char buffer[1024];
        while (ctx.running.load()) {
            ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes <= 0) {
                std::cout << "[TcpServer] Client disconnected.\n";
                break;
            }
            buffer[bytes] = '\0';

            std::string input(buffer);
            input.erase(input.find_last_not_of(" \r\n") + 1);  // trim

            std::cout << "[TcpServer] Received: " << input << "\n";

            std::string upper = input;
            std::transform(upper.begin(), upper.end(), upper.begin(),
                        [](unsigned char c){ return std::toupper(c); });

            // ===== 상태 조회: 큐에 넣지 않고 즉시 응답 =====
            // 응답 형식: STATUS|<robot_state>|<q0_deg>|...|<q12_deg>|<speed>|<pause_valid>|<pause_id>|<pause_bar>
            // q 값은 "마지막으로 명령된 목표 자세"이며 실측값이 아님. (단위: degree)
            // pause_valid=1 이면 PAUSE로 저장된 재개 지점이 있음 (RESUME 가능).
            if (upper == "GET_STATUS") {
                std::ostringstream oss;
                oss << "STATUS|" << state_to_string(ctx.robot_state.load());
                {
                    std::lock_guard<std::mutex> lk(ctx.last_q_mutex);
                    oss << std::fixed << std::setprecision(2);
                    for (double q_rad : ctx.last_q_target_snapshot) {
                        oss << "|" << (q_rad * 180.0 / M_PI);
                    }
                }
                oss << "|" << ctx.play_speed_scale;
                {
                    std::lock_guard<std::mutex> lk(ctx.play_mutex);
                    if (ctx.pause_point.valid) {
                        oss << "|1|" << ctx.pause_point.play_id << "|" << ctx.pause_point.bar;
                    } else {
                        oss << "|0|-|0";
                    }
                }
                oss << "\n";
                std::string resp = oss.str();
                send(client_fd, resp.c_str(), resp.size(), 0);
                continue;   // 큐 push / QUIT 처리 건너뜀
            }

            // 명령을 CommandQueue 에 push
            if (!input.empty()) {
                ParsedCommand parsed = command_parser.parse(input);
                command_queue.push(parsed);
            }

            if (upper == "QUIT" || upper == "Q") {
                quitting = true;
                break;
            }
        }

        close(client_fd);
    }

    ctx.running = false;
    std::cout << "[TcpServer] 스레드 종료\n";
}

void TcpServer::stop() {
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
}

void TcpServer::set_server() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "[TcpServer] socket() failed" << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (bind(server_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[TcpServer] bind() failed" << std::endl;
        return;
    }

    if (listen(server_fd_, 5) < 0) {
        std::cerr << "[TcpServer] listen() failed" << std::endl;
        return;
    }
}
