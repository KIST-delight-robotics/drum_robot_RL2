#include "hardware/robot.hpp"

Robot::Robot() {
    
}

Robot::~Robot() {
    if (dxl_port) {
        dxl_port->closePort();
        delete dxl_port;
        dxl_port = nullptr;
    }
}

void Robot::initialize() {
    can.resetCanPorts();
    can.initialize();
    init_motor_from_json();
    set_motors_socket();
    maxon_motor_setting();
    can.setSocketNonBlock();
    set_zero_tmotor();
    maxon_motor_enable();
    set_maxon_motor_mode("CSP");
    init_dynamixel();
    // TODO: USBIO
    // TODO: 아두이노
}

void Robot::init_motor_from_json() {
    using json = nlohmann::json;

    std::ifstream f("drumrobot_server/config/motors.json");
    if (!f.is_open()) {
        std::cerr << "[Robot] Failed to open config/motors.json\n";
        return;
    }

    json config = json::parse(f);

    for (auto &m : config["motors"]) {
        std::string type = m["type"];
        int id = m["id"];
        joint_names[id] = m["name"].get<std::string>();

        if (type == "TMotor") {
            auto motor = std::make_shared<TMotor>(id);
            motor->name = m["name"];
            motor->direction_sign = m["direction_sign"];
            motor->initial_joint_angle = m["initial_joint_angle"].get<double>() * M_PI / 180.0;
            motor->min_angle = m["min_angle"].get<double>() * M_PI / 180.0;
            motor->max_angle = m["max_angle"].get<double>() * M_PI / 180.0;
            motor->node_id = std::stoul(m["node_id"].get<std::string>(), nullptr, 16);
            motor->model = m["model"];
            motor->gear_ratio = m["gear_ratio"];
            motor->current_limit = m["current_limit"];
            motor->control_gain = m["control_gain"];
            motors[id] = motor;

            // std::cout << "[Robot] motor setting: " << motor->name << "\n";
        } else if (type == "MaxonMotor") {
            auto motor = std::make_shared<MaxonMotor>(id);
            motor->name = m["name"];
            motor->direction_sign = m["direction_sign"];
            motor->initial_joint_angle = m["initial_joint_angle"].get<double>() * M_PI / 180.0;
            motor->min_angle = m["min_angle"].get<double>() * M_PI / 180.0;
            motor->max_angle = m["max_angle"].get<double>() * M_PI / 180.0;
            motor->node_id = std::stoul(m["node_id"].get<std::string>(), nullptr, 16);
            motor->can_send_id = 0x600 + motor->node_id;
            motor->can_receive_id = 0x580 + motor->node_id;
            motor->tx_pdo_ids[0] = std::stoul(m["tx_pdo_ids[0]"].get<std::string>(), nullptr, 16);  // Controlword
            motor->tx_pdo_ids[1] = std::stoul(m["tx_pdo_ids[1]"].get<std::string>(), nullptr, 16);  // Target Position
            motor->tx_pdo_ids[2] = std::stoul(m["tx_pdo_ids[2]"].get<std::string>(), nullptr, 16);  // Target Velocity
            motor->tx_pdo_ids[3] = std::stoul(m["tx_pdo_ids[3]"].get<std::string>(), nullptr, 16);  // TargetTorque
            motor->rx_pdo_ids[0] = std::stoul(m["rx_pdo_ids[0]"].get<std::string>(), nullptr, 16);  // Statusword, Actual Position, Actual Torque
            motor->rx_pdo_ids[1] = std::stoul(m["rx_pdo_ids[1]"].get<std::string>(), nullptr, 16);  // ActualPosition, ActualTorque
            motor->control_kp = m["control_kp"];
            motor->control_kd = m["control_kd"];
            motor->gear_ratio = m["gear_ratio"];
            motors[id] = motor;

            // std::cout << "[Robot] motor setting: " << motor->name << "\n";
        } else if (type == "Dynamixel") {
            auto motor = std::make_shared<DynamixelMotor>(id);
            motor->name = m["name"];
            motor->direction_sign = m["direction_sign"];
            motor->initial_joint_angle = m["initial_joint_angle"].get<double>() * M_PI / 180.0;
            motor->min_angle = m["min_angle"].get<double>() * M_PI / 180.0;
            motor->max_angle = m["max_angle"].get<double>() * M_PI / 180.0;
            motor->dxl_id = m["dxl_id"];
            motors[id] = motor;

            // std::cout << "[Robot] motor setting: " << motor->name << "\n";
        }
    }
}

void Robot::set_motors_socket() {
    // 각 모터에 대해 사용 가능한 모든 CAN 소켓을 순회하며 응답 여부로 연결 확인.
    // 연결 안 된 모터는 마지막 시도 소켓 값이 남지만, 함수 끝에서 motors map에서 erase되므로 이후 사용되지 않음.
    struct can_frame frame;
    can.setSocketsTimeout(0, 10000);    // 타임아웃 10ms 설정
    can.clearReadBuffers();

    // 모든 소켓에 대해 연결을 확인
    std::map<std::string, int> sockets = can.getSocket();
    for (const auto &socket : sockets) {
        int socket_fd = socket.second;

        for (auto &[id, motor] : motors) {
            if (std::shared_ptr<TMotor> tmotor = std::dynamic_pointer_cast<TMotor>(motor)) {
                if (!tmotor->is_connected) {
                    tmotor->socket = socket_fd; // 임시로 소켓 설정
                }
            } else if (std::shared_ptr<MaxonMotor> maxon = std::dynamic_pointer_cast<MaxonMotor>(motor)) {
                if (!maxon->is_connected) {
                    maxon->socket = socket_fd;
                    m_codec.encodeCheck(maxon->can_send_id, &frame);
                    can.sendFrame(maxon->socket, frame);

                    usleep(50000);
                }
            }
        }

        // 해당 소켓으로 프레임 10개를 무작정 읽어서 버퍼에 넣음
        int readCount = 0;
        std::vector<can_frame> temp_frames;
        while (readCount < 10) {
            ssize_t result = read(socket_fd, &frame, sizeof(frame));

            if (result > 0) {
                temp_frames.push_back(frame);
            }
            readCount++;
        }

        // 버퍼에서 하나씩 읽어옴
        for (auto &[id, motor] : motors) {
            for (auto &frame : temp_frames) {
                if (std::shared_ptr<TMotor> tmotor = std::dynamic_pointer_cast<TMotor>(motor)) {
                    if ((frame.can_id & 0xFF) == tmotor->node_id) {
                        tmotor->is_connected = true;
                    }
                } else if (std::shared_ptr<MaxonMotor> maxon = std::dynamic_pointer_cast<MaxonMotor>(motor)) {
                    if (frame.can_id == maxon->can_receive_id) {
                        maxon->is_connected = true;
                    }
                }
            }
        }
    }

    // node_id 기준으로 정렬된 목록 생성
    std::vector<std::pair<uint32_t, int>> sorted_motors; // <node_id, map key>
    for (auto &[id, motor] : motors) {
        if (auto tmotor = std::dynamic_pointer_cast<TMotor>(motor)) {
            sorted_motors.push_back({tmotor->node_id, id});
        } else if (auto maxon = std::dynamic_pointer_cast<MaxonMotor>(motor)) {
            sorted_motors.push_back({maxon->node_id, id});
        }
    }
    std::sort(sorted_motors.begin(), sorted_motors.end());

    // 모터 연결 상태 확인 및 연결 안된 모터 삭제
    for (auto &[node_id, key] : sorted_motors) {
        auto it = motors.find(key);
        if (it == motors.end()) continue;

        std::shared_ptr<Motor> motor = it->second;

        if (auto tmotor = std::dynamic_pointer_cast<TMotor>(motor)) {
            if (tmotor->is_connected) {
                std::cout << "[Robot] --------------> CAN NODE ID " << node_id << " Connected. Joint " << tmotor->id << ": " << tmotor->name << "\n";
            } else {
                std::cerr << "[Robot] CAN NODE ID " << node_id << " Not Connected. Joint " << tmotor->id << ": " << tmotor->name << "\n";
                motors.erase(it);
            }
        } else if (auto maxon = std::dynamic_pointer_cast<MaxonMotor>(motor)) {
            if (maxon->is_connected) {
                std::cout << "[Robot] --------------> CAN NODE ID " << node_id << " Connected. Joint " << maxon->id << ": " << maxon->name << "\n";
            } else {
                std::cerr << "[Robot] CAN NODE ID " << node_id << " Not Connected. Joint " << maxon->id << ": " << maxon->name << "\n";
                motors.erase(it);
            }
        }
    }
}

void Robot::maxon_motor_setting() {
    // Count Maxon Motor Sockets
    for (auto &[id, motor] : motors) {
        auto maxon = std::dynamic_pointer_cast<MaxonMotor>(motor);
        if (!maxon) continue;

        if (virtual_maxon_motor.size() == 0) {
            virtual_maxon_motor.push_back(maxon);
        } else {
            bool otherSocket = true;
            int n = virtual_maxon_motor.size();
            for(int i = 0; i < n; i++) {
                if (virtual_maxon_motor[i]->socket == maxon->socket) {
                    otherSocket = false;
                }
            }

            if (otherSocket) {
                virtual_maxon_motor.push_back(maxon);
            }
        }
    }

    struct can_frame frame;
    can.setSocketsTimeout(2, 0);

    for (auto &[id, motor] : motors) {
        auto maxon = std::dynamic_pointer_cast<MaxonMotor>(motor);
        if (!maxon) continue;

        // CSP Settings
        m_codec.encodeCSPMode(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodePosOffset(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodeTorqueOffset(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        // CSV Settings
        m_codec.encodeCSVMode(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodeVelOffset(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        // CST Settings
        m_codec.encodeCSTMode(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodeTorqueOffset(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        // HMM Settings
        m_codec.encodeHomeMode(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodeHomingMethodRight(maxon->can_send_id, &frame);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodeHomeOffsetDistance(maxon->can_send_id, &frame, 0);
        can.sendandReceiveFrame(maxon->socket, frame);

        m_codec.encodeHomePosition(maxon->can_send_id, &frame, 0);
        can.sendandReceiveFrame(maxon->socket, frame);

        // m_codec.encodeCurrentThresholdLeft(maxon->can_send_id, &frame);
        // can.sendandReceiveFrame(maxon->socket, frame);
    }
}

void Robot::set_zero_tmotor() {
    struct can_frame frame;

    for (auto &[id, motor] : motors) {
        auto tmotor = std::dynamic_pointer_cast<TMotor>(motor);
        if (!tmotor) continue;

        t_codec.encodeSetOrigin(tmotor->node_id, &frame, 0);
        can.sendFrame(tmotor->socket, frame);

        usleep(100000);    // 100ms
    }

    std::cout << "[Robot] TMotor Set Zero\n";
    sleep(2);   // Set Zero 명령이 확실히 실행된 후 종료
}

void Robot::maxon_motor_enable() {
    struct can_frame frame;
    can.setSocketsTimeout(2, 0);

    // 제어 모드 설정
    for (auto &[id, motor] : motors) {
        auto maxon = std::dynamic_pointer_cast<MaxonMotor>(motor);
        if (!maxon) continue;

        m_codec.encodeHomeMode(maxon->can_send_id, &frame);
        can.sendFrame(maxon->socket, frame);

        m_codec.encodeEnterOperational(maxon->node_id, &frame);
        can.sendFrame(maxon->socket, frame);

        usleep(100000);

        m_codec.encodeShutdown(maxon->tx_pdo_ids[0], &frame);
        can.sendFrame(maxon->socket, frame);

        m_codec.encodeSync(&frame);
        can.sendFrame(maxon->socket, frame);

        usleep(100000);
        
        m_codec.encodeEnable(maxon->tx_pdo_ids[0], &frame);
        can.sendFrame(maxon->socket, frame);

        m_codec.encodeSync(&frame);
        can.sendFrame(maxon->socket, frame);

        usleep(100000);

        // 현재 사용하지 않는 기능
        // m_codec.encodeStartHoming(maxon->tx_pdo_ids[0], &frame);
        // can.sendFrame(maxon->socket, frame);

        // m_codec.encodeSync(&frame);
        // can.sendFrame(maxon->socket, frame);

        // usleep(100000);
    }

    std::cout << "[Robot] Maxon Motor Enable\n";
}

void Robot::set_maxon_motor_mode(const std::string& targetMode) {
    struct can_frame frame;
    can.setSocketsTimeout(0, 10000);

    for (auto &[id, motor] : motors) {
        auto maxon = std::dynamic_pointer_cast<MaxonMotor>(motor);
        if (!maxon) continue;

        if (targetMode == "CSV") {          // Cyclic Sync Velocity Mode
            m_codec.encodeCSVMode(maxon->can_send_id, &frame);
            can.sendFrame(maxon->socket, frame);
        } else if (targetMode == "CST") {   // Cyclic Sync Torque Mode
            m_codec.encodeCSTMode(maxon->can_send_id, &frame);
            can.sendFrame(maxon->socket, frame);
        } else if (targetMode == "HMM") {   // Homming Mode
            m_codec.encodeHomeMode(maxon->can_send_id, &frame);
            can.sendFrame(maxon->socket, frame);
        } else if (targetMode == "CSP") {   // Cyclic Sync Position Mode
            m_codec.encodeCSPMode(maxon->can_send_id, &frame);
            can.sendFrame(maxon->socket, frame);
        }

        // 모드 바꾸고 껐다 켜주기

        m_codec.encodeShutdown(maxon->tx_pdo_ids[0], &frame);
        can.sendFrame(maxon->socket, frame);

        m_codec.encodeSync(&frame);
        can.sendFrame(maxon->socket, frame);

        usleep(100);

        m_codec.encodeEnable(maxon->tx_pdo_ids[0], &frame);
        can.sendFrame(maxon->socket, frame);

        m_codec.encodeSync(&frame);
        can.sendFrame(maxon->socket, frame);

        usleep(100);
    }

    std::cout << "[Robot] Maxon Motor Mode: " << targetMode << "\n";
}

void Robot::init_dynamixel() {
    dynamixel::PortHandler *port;
    dynamixel::PacketHandler *pkt;

    port = dynamixel::PortHandler::getPortHandler(DXL_PORT);
    pkt = dynamixel::PacketHandler::getPacketHandler(2.0);

    // Open port
    if (port->openPort())
    {
        printf("[Robot] -------------- Open Dynamixel Port\n");
        set_dxl_latency(DXL_PORT, 1);
    }
    else
    {
        printf("[Robot] Failed to open the Dynamixel port!\n");

        // 다이나믹셀 모터 삭제
        std::vector<int> to_remove;

        for (auto &[id, motor] : motors) {
            auto dxl = std::dynamic_pointer_cast<DynamixelMotor>(motor);
            if (dxl) {
                to_remove.push_back(id);
            }
        }

        for (int id : to_remove) {
            motors.erase(id);
        }

        return;
    }

    // Set port baudrate
    if (port->setBaudRate(4500000))
    {
        printf("[Robot] -------------- change the baudrate!\n");
    }
    else
    {
        printf("\n[Robot] Failed to change the baudrate!\n");

        // 다이나믹셀 모터 삭제
        std::vector<int> to_remove;

        for (auto &[id, motor] : motors) {
            auto dxl = std::dynamic_pointer_cast<DynamixelMotor>(motor);
            if (dxl) {
                to_remove.push_back(id);
            }
        }

        for (int id : to_remove) {
            motors.erase(id);
        }

        return;
    }

    // 모터 연결 확인 및 연결 안된 모터 삭제
    std::vector<int> to_remove;

    uint8_t err = 0;
    for (auto &[id, motor] : motors) {
        auto dxl = std::dynamic_pointer_cast<DynamixelMotor>(motor);
        if (!dxl) continue;

        uint16_t dxl_model_number = 0;
        uint8_t dxl_error = 0;

        int dxl_comm_result = pkt->ping(port, dxl->dxl_id, &dxl_model_number, &dxl_error);

        if (dxl_comm_result == COMM_SUCCESS && dxl_error == 0) {
            // DXL 토크 ON
            pkt->write1ByteTxRx(port, dxl->dxl_id, 64, 1, &err);
            printf("[Robot] --------------> ID:%03d Found! Model number: %d\n", dxl->dxl_id, dxl_model_number);
        } else {
            to_remove.push_back(id);
            printf("[Robot] ID:%03d COMM FAIL\n", dxl->dxl_id);
        }
    }

    for (int id : to_remove) {
        motors.erase(id);
    }

    // group sync 생성
    bool has_dxl = false;
    for (auto &[id, motor] : motors) {
        if (std::dynamic_pointer_cast<DynamixelMotor>(motor)) {
            has_dxl = true;
            break;
        }
    }
    if (!has_dxl) {
        return;  // group sync 생성 안 함
    }

    dxl_sync_write = std::make_unique<dynamixel::GroupSyncWrite>(port, pkt, 108, 12);
    dxl_sync_read = std::make_unique<dynamixel::GroupSyncRead>(port, pkt, 132, 4);
    for (auto &[id, motor] : motors) {
        auto dxl = std::dynamic_pointer_cast<DynamixelMotor>(motor);
        if (!dxl) continue;
        // read는 한 번만 등록하면 끝  
        if (!dxl_sync_read->addParam(dxl->dxl_id)) {
            std::cerr << "[Robot] dxl_sync_read addParam failed for ID:"
                    << (int)dxl->dxl_id << "\n";
            continue;
        }
    }
    dxl_port = port;
    dxl_packet  = pkt;

    // 초기 위치로 보내기
    set_dxl_initial_pose();
}

void Robot::set_dxl_latency(const std::string &dev_path, int latency_ms) {
    // latency_timer 값 바꿔주기
    auto pos = dev_path.find_last_of('/');
    std::string dev = (pos == std::string::npos) ? dev_path : dev_path.substr(pos + 1);

    std::string sysfs =
        "/sys/bus/usb-serial/devices/" + dev + "/latency_timer";

    std::ofstream ofs(sysfs);
    if (!ofs) {
        std::cerr << "[Robot] latency_timer open 실패: "
                  << sysfs << "\n";
        return;
    }
    ofs << latency_ms;
    if (!ofs) {
        std::cerr << "[Robot] latency_timer 쓰기 실패: " << sysfs << "\n";
    } else {
        std::cout << "[Robot] " << dev << " latency_timer = "
                  << latency_ms << "ms 설정 완료\n";
    }
}

void Robot::set_dxl_initial_pose() {
    // 다이나믹셀은 초기 위치가 고정되어 있지 않음
    const int32_t total_time = 2000;  // 이동 시간 [ms]

    dxl_sync_write->clearParam();
    for (auto &[id, motor] : motors) {
        auto dxl = std::dynamic_pointer_cast<DynamixelMotor>(motor);
        if (!dxl) continue;

        double motor_position = dxl->joint_angle_to_motor_position(0.0);    // 초기 위치 0

        int32_t values[3];
        uint8_t param[12];
        // Profile Acceleration, Profile Velocity, Goal Position
        values[0] = total_time;
        values[1] = total_time / 2;
        values[2] = dxl->angle_to_tick(motor_position);
        memcpy(param, values, sizeof(values));

        if (!dxl_sync_write->addParam(dxl->dxl_id, param)) {
            std::cerr << "[Robot] init pose: dxl_sync_write addParam failed for ID:"
                      << (int)dxl->dxl_id << "\n";
            continue;
        }

        dxl->current_position = motor_position;
        dxl->current_joint_angle = 0.0;
    }

    if (dxl_sync_write->txPacket() != COMM_SUCCESS) {
        std::cerr << "[Robot] init pose: SyncWrite failed\n";
    } else {
        printf("[Robot] -------------- Dynamixel moving to initial pose\n");
    }

    // 모터가 초기 위치에 도착할 때까지 대기
    usleep((total_time + 200) * 1000);
}
