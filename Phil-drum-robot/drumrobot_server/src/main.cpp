#include <iostream>
#include <thread>
#include <pthread.h>
#include <map>
#include <string>
#include <memory>

#include "common/app_context.hpp"
#include "common/command_queue.hpp"
#include "common/control_queue.hpp"
#include "common/motion_queue.hpp"
#include "hardware/robot.hpp"
#include "tcp/tcp_server.hpp"
#include "realtime/controller.hpp"
#include "trajectory/motion_planner.hpp"

#define PORT 1951   // Phil Collins 출생연도

void set_priority(std::thread& t, int priority) {
    // 값이 클수록 더 높은 우선순위
    sched_param sch;
    sch.sched_priority = priority;
    if (pthread_setschedparam(t.native_handle(), SCHED_FIFO, &sch) != 0) {
        std::cerr << "Failed to set priority " << priority << std::endl;
    }
}

int main(int argc, char* argv[]) {
    AppContext ctx;
    CommandQueue command_queue;
    ControlQueue control_queue;
    MotionQueue motion_queue;

    Robot robot;
    robot.initialize();

    AudioPlayer audio_player;
    audio_player.initialize();
    
    TcpServer server(ctx, PORT, command_queue);
    Controller controller(ctx, control_queue, robot, audio_player);
    MotionPlanner motion_planner(ctx, command_queue, control_queue, motion_queue, robot, audio_player);

    std::thread send_thread(&Controller::send_loop, &controller);
    std::thread recv_thread(&Controller::recv_loop, &controller);
    std::thread motion_planning_thread(&MotionPlanner::run, &motion_planner);
    std::thread tcp_server_thread(&TcpServer::run, &server);

    set_priority(send_thread, 40);
    set_priority(recv_thread, 30);
    set_priority(motion_planning_thread, 20);
    set_priority(tcp_server_thread, 10);

    send_thread.join();
    recv_thread.join();
    motion_planning_thread.join();
    tcp_server_thread.join();
    server.stop();

    return 0;
}