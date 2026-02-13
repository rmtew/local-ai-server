/*
 * Multipart Parser
 * Binary-safe multipart/form-data parser with zero-copy part extraction.
 */

#ifndef LOCAL_AI_MULTIPART_H
#define LOCAL_AI_MULTIPART_H

#include <stddef.h>

#define MULTIPART_MAX_PARTS 16

/* A single part from a multipart/form-data body */
typedef struct {
    char name[128];             /* Content-Disposition "name" parameter */
    char filename[256];         /* Content-Disposition "filename" parameter (empty if none) */
    char content_type[128];     /* Part Content-Type (empty if none) */
    const unsigned char *data;  /* Pointer into original body buffer (zero-copy) */
    size_t data_len;            /* Length of part data */
} MultipartPart;

/* Extract boundary string from Content-Type header value.
 * Writes boundary into out_boundary (including leading "--").
 * Returns boundary length, or 0 on error. */
size_t multipart_get_boundary(const char *content_type, char *out_boundary, size_t out_size);

/* Parse a multipart/form-data body.
 * boundary: the boundary string (with leading "--")
 * boundary_len: length of boundary string
 * body: raw body data
 * body_len: length of body data
 * parts: output array of parsed parts
 * max_parts: capacity of parts array
 * Returns: number of parts parsed, or -1 on error. */
int multipart_parse(const char *boundary, size_t boundary_len,
                    const unsigned char *body, size_t body_len,
                    MultipartPart *parts, int max_parts);

/* Find a part by name. Returns pointer to the part, or NULL. */
const MultipartPart *multipart_find(const MultipartPart *parts, int count, const char *name);

#endif /* LOCAL_AI_MULTIPART_H */
