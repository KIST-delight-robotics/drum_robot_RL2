#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

class Logger {
public:
    Logger(const std::string& name);
    ~Logger();

    void set_header(const std::vector<std::string>& columns);
    void record(const std::vector<double>& values);
    void record(const std::vector<std::string>& values);
    void record(const std::array<double, 13>& values);

private:
    std::ofstream file;
    std::chrono::steady_clock::time_point start;
    const std::string base_path = "drumrobot_server/log/";

    std::string make_filename(const std::string& name);
};
 