/*
 * HTTP Layer
 * Single-threaded HTTP server: Winsock2 on Windows, POSIX sockets on Linux.
 */

#ifndef LOCAL_AI_HTTP_H
#define LOCAL_AI_HTTP_H

#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR   (-1)
#define closesocket    close
#endif

#include <stddef.h>

/* Parsed HTTP request */
typedef struct {
    char method[16];            /* GET, POST, etc. */
    char path[256];             /* Request path */
    char content_type[256];     /* Content-Type header value */
    size_t content_length;      /* Content-Length value */
    unsigned char *body;        /* Request body (malloc'd, caller frees) */
    size_t body_len;            /* Actual body length received */
    int detached;               /* If set by handler, accept loop won't close socket */
} HttpRequest;

/* Server state */
typedef struct {
    SOCKET listen_sock;
    int port;
    volatile int running;
} HttpServer;

/* Handler callback: called for each request.
 * Must send a response using http_send_response() or http_send_json_error().
 * request->body is freed by the caller after the handler returns.
 * Handler may set request->detached=1 to keep the socket open after return. */
typedef void (*http_handler_fn)(SOCKET client, HttpRequest *request, void *user_data);

/* Initialize server (WSAStartup, bind, listen).
 * Returns 0 on success, -1 on error. */
int http_server_init(HttpServer *srv, int port);

/* Run the accept loop (blocking). Calls handler for each request.
 * Returns when srv->running is set to 0. */
void http_server_run(HttpServer *srv, http_handler_fn handler, void *user_data);

/* Send an HTTP response with the given status code, content type, and body. */
void http_send_response(SOCKET client, int status, const char *content_type,
                        const char *body, size_t body_len);

/* Send a JSON error response matching OpenAI error format. */
void http_send_json_error(SOCKET client, int status, const char *message, const char *type);

/* Send SSE response headers (200 OK, text/event-stream, TCP_NODELAY). */
void http_send_sse_headers(SOCKET client);

/* Send a single SSE event: "data: <payload>\n\n". */
void http_send_sse_event(SOCKET client, const char *data, size_t len);

/* Shut down the server (close listen socket, WSACleanup). */
void http_server_shutdown(HttpServer *srv);

#endif /* LOCAL_AI_HTTP_H */
