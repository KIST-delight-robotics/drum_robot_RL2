#pragma once

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <utility>
#include <random>
#include <iomanip>

#include "nlohmann/json.hpp"

class KinematicsSolver {
public:
    KinematicsSolver();
    ~KinematicsSolver();

    void initialize();

    struct IKResult {
        std::array<double, 9> q;  // [θ0~θ8], rad
        bool success = false;   // 성공 여부
    };

    IKResult solve_ik(
        const std::array<double, 3>& pR,
        const std::array<double, 3>& pL,
        double theta0,
        double theta7,
        double theta8,
        bool print_err
    ) const;

    bool check_joint_limits(const std::array<double, 9>& q, bool print_err) const;    // q 벡터가 모든 관절 한계 내에 있는지 확인

    struct FKResult {
        std::array<double, 3> pR;   // 오른손 끝 좌표 (드럼 스틱 끝)
        std::array<double, 3> pL;   // 왼손 끝 좌표
        bool success = false;
    };
    
    FKResult solve_fk(const std::array<double, 9>& q);

private:
    struct JointLimit {
        double min_angle;   // rad
        double max_angle;   // rad
    };

    struct LinkLength {
        double waist;       // 어깨 간격
        double upper_arm;   // 상완
        double forearm;     // 하완
        double stick;       // 스틱
    };

    std::map<int, JointLimit> joint_limits;
    LinkLength link_length;

    void verify_fk_ik(int num_tests = 1000, double tolerance_deg = 0.01);

    double get_effective_length(double theta_wrist) const;  // 하완+스틱 합성 링크 길이
    double get_effective_theta(double theta_wrist) const;   // 하완+스틱 합성 링크 방향각

    std::array<std::array<double, 4>, 4> dh_transform(double a, double alpha, double d, double theta);
    void mat4_mul_inplace(std::array<std::array<double, 4>, 4>& A, const std::array<std::array<double, 4>, 4>& B);
};