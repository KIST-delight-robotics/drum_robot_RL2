#include <limits.h>
#include <string>

#include <fcntl.h>
#include <unistd.h>

std::string get_hostname();  // 현재 머신의 호스트네임 조회 (can_ports.json, audio.json 키)