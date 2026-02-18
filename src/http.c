/*
 * HTTP Layer
 * HTTP server implementation: Winsock2 on Windows, POSIX sockets on Linux.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "http.h"
#include "json.h"

#ifdef _WIN32
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define SOCK_ERRNO WSAGetLastError()
#else
#include <strings.h>
#define SOCK_ERRNO errno
#define _strnicmp strncasecmp
#define _atoi64(s) strtoll((s), NULL, 10)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <mstcpip.h>   /* SIO_TCP_SET_ACK_FREQUENCY */
#endif

#define HEADER_BUF_SIZE  (16 * 1024)
#define MAX_BODY_SIZE    (100 * 1024 * 1024)  /* 100 MB */

/* Find end of HTTP headers (\r\n\r\n) in buffer.
 * Returns pointer past the \r\n\r\n, or NULL if not found. */
static char *find_header_end(char *buf, int len) {
    for (int i = 0; i + 3 < len; i++) {
        if (buf[i] == '\r' && buf[i+1] == '\n' &&
            buf[i+2] == '\r' && buf[i+3] == '\n') {
            return buf + i + 4;
        }
    }
    return NULL;
}

/* Case-insensitive header search. Returns pointer to header value (after ": "),
 * or NULL if not found. The returned pointer is into the header buffer. */
static const char *find_header_value(const char *headers, const char *name) {
    size_t name_len = strlen(name);
    const char *p = headers;
    while (*p) {
        if (_strnicmp(p, name, name_len) == 0 && p[name_len] == ':') {
            const char *val = p + name_len + 1;
            while (*val == ' ' || *val == '\t') val++;
            return val;
        }
        /* Advance to next line */
        while (*p && *p != '\n') p++;
        if (*p == '\n') p++;
    }
    return NULL;
}

/* Copy header value until CR/LF/NUL into dest (with size limit). */
static void copy_header_value(char *dest, size_t dest_size, const char *src) {
    size_t i = 0;
    while (i < dest_size - 1 && src[i] && src[i] != '\r' && src[i] != '\n') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
}

/* Parse the request line: "METHOD /path HTTP/1.1\r\n" */
static int parse_request_line(const char *buf, HttpRequest *req) {
    /* Method */
    const char *p = buf;
    int i = 0;
    while (*p && *p != ' ' && i < (int)sizeof(req->method) - 1) {
        req->method[i++] = *p++;
    }
    req->method[i] = '\0';
    if (*p != ' ') return -1;
    p++;

    /* Path */
    i = 0;
    while (*p && *p != ' ' && *p != '?' && i < (int)sizeof(req->path) - 1) {
        req->path[i++] = *p++;
    }
    req->path[i] = '\0';

    return 0;
}

int http_server_init(HttpServer *srv, int port) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "WSAStartup failed: %d\n", WSAGetLastError());
        return -1;
    }
#endif

    srv->listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (srv->listen_sock == INVALID_SOCKET) {
        fprintf(stderr, "socket() failed: %d\n", SOCK_ERRNO);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    /* Allow port reuse */
    int opt = 1;
    setsockopt(srv->listen_sock, SOL_SOCKET, SO_REUSEADDR,
               (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(srv->listen_sock, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "bind() failed on port %d: %d\n", port, SOCK_ERRNO);
        closesocket(srv->listen_sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    if (listen(srv->listen_sock, 1) == SOCKET_ERROR) {
        fprintf(stderr, "listen() failed: %d\n", SOCK_ERRNO);
        closesocket(srv->listen_sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    srv->port = port;
    srv->running = 1;
    return 0;
}

void http_server_run(HttpServer *srv, http_handler_fn handler, void *user_data) {
    while (srv->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        SOCKET client = accept(srv->listen_sock, (struct sockaddr *)&client_addr, &addr_len);
        if (client == INVALID_SOCKET) {
            if (!srv->running) break;
            fprintf(stderr, "accept() failed: %d\n", SOCK_ERRNO);
            continue;
        }

        /* Disable Nagle's algorithm — send small packets immediately.
         * Critical for HTTP responses: avoids 200ms coalescing delays
         * when header and body are sent as separate send() calls. */
        {
            int flag = 1;
            setsockopt(client, IPPROTO_TCP, TCP_NODELAY,
                       (const char *)&flag, sizeof(flag));
        }

#ifdef _WIN32
        /* Reduce delayed-ACK from default 200ms to 1ms.
         * Windows TCP stacks wait before ACK-ing, adding round-trip latency
         * on localhost where RTT is effectively 0. */
        {
            DWORD freq = 1;
            DWORD bytes_returned = 0;
            WSAIoctl(client, SIO_TCP_SET_ACK_FREQUENCY,
                     &freq, sizeof(freq), NULL, 0,
                     &bytes_returned, NULL, NULL);
        }
#endif

        /* Receive headers */
        char header_buf[HEADER_BUF_SIZE];
        int header_total = 0;
        char *header_end = NULL;

        while (header_total < HEADER_BUF_SIZE - 1) {
            int n = recv(client, header_buf + header_total,
                         HEADER_BUF_SIZE - 1 - header_total, 0);
            if (n <= 0) break;
            header_total += n;
            header_buf[header_total] = '\0';
            header_end = find_header_end(header_buf, header_total);
            if (header_end) break;
        }

        if (!header_end) {
            http_send_json_error(client, 400, "Invalid request: no header terminator", "invalid_request_error");
            closesocket(client);
            continue;
        }

        /* Parse request */
        HttpRequest req;
        memset(&req, 0, sizeof(req));

        if (parse_request_line(header_buf, &req) != 0) {
            http_send_json_error(client, 400, "Malformed request line", "invalid_request_error");
            closesocket(client);
            continue;
        }

        /* Extract headers */
        const char *ct = find_header_value(header_buf, "Content-Type");
        if (ct) copy_header_value(req.content_type, sizeof(req.content_type), ct);

        const char *cl = find_header_value(header_buf, "Content-Length");
        if (cl) req.content_length = (size_t)_atoi64(cl);

        /* Read body if present */
        if (req.content_length > 0) {
            if (req.content_length > MAX_BODY_SIZE) {
                http_send_json_error(client, 413, "Request body too large (max 100MB)", "invalid_request_error");
                closesocket(client);
                continue;
            }

            req.body = (unsigned char *)malloc(req.content_length);
            if (!req.body) {
                http_send_json_error(client, 500, "Out of memory", "server_error");
                closesocket(client);
                continue;
            }

            /* Copy any body bytes already read with headers */
            size_t already = (size_t)(header_buf + header_total - (char *)header_end);
            if (already > req.content_length) already = req.content_length;
            if (already > 0) {
                memcpy(req.body, header_end, already);
            }
            req.body_len = already;

            /* Read remaining body */
            while (req.body_len < req.content_length) {
                int n = recv(client, (char *)req.body + req.body_len,
                             (int)(req.content_length - req.body_len), 0);
                if (n <= 0) break;
                req.body_len += (size_t)n;
            }
        }

        /* Dispatch to handler */
        handler(client, &req, user_data);

        /* Cleanup */
        free(req.body);
        if (!req.detached)
            closesocket(client);
    }
}

void http_send_response(SOCKET client, int status, const char *content_type,
                        const char *body, size_t body_len) {
    const char *status_text = "OK";
    if (status == 400) status_text = "Bad Request";
    else if (status == 404) status_text = "Not Found";
    else if (status == 405) status_text = "Method Not Allowed";
    else if (status == 413) status_text = "Payload Too Large";
    else if (status == 500) status_text = "Internal Server Error";

    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);

    send(client, header, hlen, 0);
    if (body && body_len > 0) {
        size_t sent = 0;
        while (sent < body_len) {
            int chunk = (int)(body_len - sent);
            if (chunk > 65536) chunk = 65536;
            int n = send(client, body + sent, chunk, 0);
            if (n <= 0) break;
            sent += (size_t)n;
        }
    }
}

void http_send_json_error(SOCKET client, int status, const char *message, const char *type) {
    char buf[1024];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));
    jw_object_start(&w);
    jw_field_object_start(&w, "error");
    jw_field_string(&w, "message", message);
    jw_field_string(&w, "type", type);
    jw_object_end(&w);
    jw_object_end(&w);

    http_send_response(client, status, "application/json", buf, jw_length(&w));
}

void http_send_sse_headers(SOCKET client) {
    /* TCP_NODELAY already set on accept — no need to set again */
    const char *headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send(client, headers, (int)strlen(headers), 0);
}

void http_send_sse_event(SOCKET client, const char *data, size_t len) {
    /* "data: " + payload + "\n\n" */
    send(client, "data: ", 6, 0);
    if (len > 0) {
        size_t sent = 0;
        while (sent < len) {
            int chunk = (int)(len - sent);
            if (chunk > 65536) chunk = 65536;
            int n = send(client, data + sent, chunk, 0);
            if (n <= 0) break;
            sent += (size_t)n;
        }
    }
    send(client, "\n\n", 2, 0);
}

void http_server_shutdown(HttpServer *srv) {
    srv->running = 0;

    /* Self-connect to unblock accept() */
    SOCKET wake = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (wake != INVALID_SOCKET) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons((uint16_t)srv->port);
        connect(wake, (struct sockaddr *)&addr, sizeof(addr));
        closesocket(wake);
    }

    closesocket(srv->listen_sock);
#ifdef _WIN32
    WSACleanup();
#endif
}
