#include "util/logger.hpp"

Logger::Logger(const std::string& name) {
    std::string filename = make_filename(name);
    file.open(filename);
    if (!file.is_open()) {
        std::cerr << "[Logger] 파일 열기 실패: " << filename << "\n";
    } else {
        std::cout << "[Logger] " << name << " 로그 기록 시작\n";
    }

    start = std::chrono::steady_clock::now();
}

Logger::~Logger() {
    if (file.is_open()) file.close();
}

void Logger::set_header(const std::vector<std::string> &columns) {
    file << "t" << ",";

    for (size_t i = 0; i < columns.size(); ++i) {
        file << columns[i];
        if (i + 1 < columns.size()) file << ",";
    }
    file << "\n";
}

void Logger::record(const std::vector<double>& values) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start);

    file << std::fixed << std::setprecision(4) << elapsed.count() << ",";

    for (size_t i = 0; i < values.size(); ++i) {
        file << values[i];
        if (i + 1 < values.size()) file << ",";
    }
    file << "\n";
}

void Logger::record(const std::vector<std::string>& values) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start);

    file << std::fixed << std::setprecision(4) << elapsed.count() << ",";

    for (size_t i = 0; i < values.size(); ++i) {
        file << values[i];
        if (i + 1 < values.size()) file << ",";
    }
    file << "\n";
}

void Logger::record(const std::array<double, 13>& values) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start);

    file << std::fixed << std::setprecision(4) << elapsed.count() << ",";

    for (size_t i = 0; i < 13; ++i) {
        file << values[i];
        if (i + 1 < values.size()) file << ",";
    }
    file << "\n";
}

std::string Logger::make_filename(const std::string& name) {
    std::time_t now = std::time(nullptr);
    std::tm *t = std::localtime(&now);

    std::ostringstream oss;
    oss << base_path << "log_"
        << std::setfill('0')
        << std::setw(2) << t->tm_mon + 1
        << std::setw(2) << t->tm_mday << "_"
        << std::setw(2) << t->tm_hour
        << std::setw(2) << t->tm_min
        << "_" << name << ".csv";
    return oss.str();
}
