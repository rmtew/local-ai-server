/*
 * HTTP Layer
 * Winsock2-based HTTP server for single-threaded request handling.
 */

#ifndef LOCAL_AI_HTTP_H
#define LOCAL_AI_HTTP_H

#include <winsock2.h>
#include <stddef.h>

/* Parsed HTTP request */
typedef struct {
    char method[16];            /* GET, POST, etc. */
    char path[256];             /* Request path */
    char content_type[256];     /* Content-Type header value */
    size_t content_length;      /* Content-Length value */
    unsigned char *body;        /* Request body (malloc'd, caller frees) */
    size_t body_len;            /* Actual body length received */
} HttpRequest;

/* Server state */
typedef struct {
    SOCKET listen_sock;
    int port;
    volatile int running;
} HttpServer;

/* Handler callback: called for each request.
 * Must send a response using http_send_response() or http_send_json_error().
 * request->body is freed by the caller after the handler returns. */
typedef void (*http_handler_fn)(SOCKET client, const HttpRequest *request, void *user_data);

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

/* Shut down the server (close listen socket, WSACleanup). */
void http_server_shutdown(HttpServer *srv);

#endif /* LOCAL_AI_HTTP_H */
