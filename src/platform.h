/*
 * platform.h - Cross-platform timing and memory abstraction
 *
 * Provides platform_time_ms() using QPC on Windows, clock_gettime on Linux.
 * Provides platform_rss_bytes() for process resident memory measurement.
 */

#ifndef LOCAL_AI_PLATFORM_H
#define LOCAL_AI_PLATFORM_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

static inline double platform_time_ms(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Returns process resident set size (working set) in bytes. */
static inline size_t platform_rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize;
    return 0;
}

#else
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline double platform_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* Returns process resident set size (VmRSS) in bytes. */
static inline size_t platform_rss_bytes(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            rss = (size_t)atol(line + 6) * 1024;
            break;
        }
    }
    fclose(f);
    return rss;
}

#endif

#endif /* LOCAL_AI_PLATFORM_H */
