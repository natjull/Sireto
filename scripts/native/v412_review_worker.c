#include <CommonCrypto/CommonDigest.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static int write_all(int fd, const void *data, size_t size) {
    const unsigned char *bytes = data; size_t offset = 0;
    while (offset < size) { ssize_t n = write(fd, bytes + offset, size - offset); if (n <= 0) return -1; offset += (size_t)n; }
    return 0;
}

static unsigned char *read_all(int fd, size_t *size) {
    unsigned char *data = malloc(65537); if (!data) return NULL; size_t used = 0;
    while (used <= 65536) { ssize_t n = read(fd, data + used, 65537 - used); if (n < 0) { free(data); return NULL; } if (!n) break; used += (size_t)n; }
    if (used > 65536) { free(data); return NULL; } *size = used; return data;
}

static int parse_fd(const char *text) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > INT32_MAX) return -1;
    return (int)value;
}

int main(int argc, char **argv) {
    if (argc != 6) return 64;
    int input = parse_fd(argv[2]), output = parse_fd(argv[3]);
    int ready = parse_fd(argv[4]), gate = parse_fd(argv[5]);
    if (input < 0 || output < 0 || ready < 0 || gate < 0) return 65;
    if (write_all(ready, "R", 1)) return 66;
    char go = 0; if (read(gate, &go, 1) != 1 || go != 'G') return 67;
    close(ready); close(gate);
    size_t size = 0; unsigned char *data = read_all(input, &size); if (!data) return 68;
    unsigned char result[68]; size_t result_size = 0;
    if (!strcmp(argv[1], "IDENTITY_DISCOVERY")) {
        CC_SHA256(data, (CC_LONG)size, result); result_size = 32;
    } else if (!strcmp(argv[1], "FROZEN_CANDIDATE_COMPARISON")) {
        if (size < 36) { free(data); return 69; }
        memcpy(result, data, 36); CC_SHA256(data + 36, (CC_LONG)(size - 36), result + 36); result_size = 68;
    } else if (!strcmp(argv[1], "CAPABILITY_PROBE")) {
        char *path = calloc(size + 1, 1); if (!path) { free(data); return 70; } memcpy(path, data, size);
        unsigned char mask = 0;
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) mask |= 1;
        else {
            struct sockaddr_in address = {0};
            address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (bind(sock, (const struct sockaddr *)&address, sizeof(address)) < 0) mask |= 1;
            close(sock);
        }
        pid_t child = fork(); if (child < 0) mask |= 2; else { if (!child) _exit(0); waitpid(child, NULL, 0); }
        pid_t spawned = 0; char *args[] = {"/usr/bin/true", NULL}; if (posix_spawn(&spawned, args[0], NULL, NULL, args, environ)) mask |= 4; else waitpid(spawned, NULL, 0);
        int fd = open(path, O_RDONLY); if (fd < 0) mask |= 8; else close(fd);
        fd = open(path, O_WRONLY); if (fd < 0) mask |= 16; else close(fd);
        pid_t reexecuted = 0; char *self_args[] = {argv[0], NULL};
        if (posix_spawn(&reexecuted, self_args[0], NULL, NULL, self_args, environ)) mask |= 32;
        else waitpid(reexecuted, NULL, 0);
        result[0] = mask; result_size = 1; free(path);
    } else { free(data); return 71; }
    free(data); if (write_all(output, result, result_size)) return 72; return 0;
}
