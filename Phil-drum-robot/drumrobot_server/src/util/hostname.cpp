#include "util/hostname.hpp"

std::string get_hostname() {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("[hostname] gethostname");
        return std::string{};
    }
    hostname[HOST_NAME_MAX] = '\0';
    return std::string(hostname);
}
