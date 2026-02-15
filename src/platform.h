/*
 * platform.h - Cross-platform timing abstraction
 *
 * Provides platform_time_ms() using QPC on Windows, clock_gettime on Linux.
 */

#ifndef LOCAL_AI_PLATFORM_H
#define LOCAL_AI_PLATFORM_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline double platform_time_ms(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

#else
#include <time.h>

static inline double platform_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

#endif

#endif /* LOCAL_AI_PLATFORM_H */
