/*
 * Multipart Parser
 * Binary-safe multipart/form-data parser.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "multipart.h"

#include <string.h>
#include <stdio.h>

/* Binary-safe memory search (like memmem on Linux).
 * Returns pointer to first occurrence of needle in haystack, or NULL. */
static const unsigned char *mem_find(const unsigned char *haystack, size_t haystack_len,
                                     const unsigned char *needle, size_t needle_len) {
    if (needle_len == 0) return haystack;
    if (needle_len > haystack_len) return NULL;

    size_t limit = haystack_len - needle_len;
    for (size_t i = 0; i <= limit; i++) {
        if (haystack[i] == needle[0] &&
            memcmp(haystack + i, needle, needle_len) == 0) {
            return haystack + i;
        }
    }
    return NULL;
}

/* Extract a quoted parameter value from a header line.
 * E.g., from 'name="file"' extracts 'file'.
 * Returns length written, or 0 if not found. */
static size_t extract_param(const char *header, const char *param,
                            char *out, size_t out_size) {
    size_t param_len = strlen(param);
    const char *p = header;
    while ((p = strstr(p, param)) != NULL) {
        p += param_len;
        if (*p != '=') continue;
        p++;
        if (*p == '"') {
            p++;
            size_t i = 0;
            while (*p && *p != '"' && i < out_size - 1) {
                out[i++] = *p++;
            }
            out[i] = '\0';
            return i;
        } else {
            /* Unquoted value -- read until ; or whitespace or end of line */
            size_t i = 0;
            while (*p && *p != ';' && *p != '\r' && *p != '\n' &&
                   *p != ' ' && i < out_size - 1) {
                out[i++] = *p++;
            }
            out[i] = '\0';
            return i;
        }
    }
    out[0] = '\0';
    return 0;
}

size_t multipart_get_boundary(const char *content_type, char *out_boundary, size_t out_size) {
    const char *bp = strstr(content_type, "boundary=");
    if (!bp) return 0;
    bp += 9; /* strlen("boundary=") */

    /* Start with "--" prefix */
    if (out_size < 3) return 0;
    out_boundary[0] = '-';
    out_boundary[1] = '-';

    size_t i = 2;
    /* Handle quoted boundary */
    if (*bp == '"') {
        bp++;
        while (*bp && *bp != '"' && i < out_size - 1) {
            out_boundary[i++] = *bp++;
        }
    } else {
        while (*bp && *bp != ';' && *bp != '\r' && *bp != '\n' &&
               *bp != ' ' && i < out_size - 1) {
            out_boundary[i++] = *bp++;
        }
    }
    out_boundary[i] = '\0';
    return i;
}

int multipart_parse(const char *boundary, size_t boundary_len,
                    const unsigned char *body, size_t body_len,
                    MultipartPart *parts, int max_parts) {
    int count = 0;
    const unsigned char *pos = body;
    const unsigned char *end = body + body_len;
    const unsigned char *bnd = (const unsigned char *)boundary;

    /* Find first boundary */
    pos = mem_find(pos, (size_t)(end - pos), bnd, boundary_len);
    if (!pos) return 0;
    pos += boundary_len;

    while (count < max_parts && pos < end) {
        /* Check for terminating boundary (--) */
        if (pos + 1 < end && pos[0] == '-' && pos[1] == '-') {
            break;
        }

        /* Skip CRLF after boundary */
        if (pos + 1 < end && pos[0] == '\r' && pos[1] == '\n') {
            pos += 2;
        }

        /* Parse part headers (until empty line \r\n\r\n) */
        const unsigned char *header_end = mem_find(pos, (size_t)(end - pos),
                                                   (const unsigned char *)"\r\n\r\n", 4);
        if (!header_end) break;

        /* Extract headers as a null-terminated string for parsing */
        size_t header_len = (size_t)(header_end - pos);
        char header_str[2048];
        if (header_len >= sizeof(header_str)) header_len = sizeof(header_str) - 1;
        memcpy(header_str, pos, header_len);
        header_str[header_len] = '\0';

        MultipartPart *part = &parts[count];
        memset(part, 0, sizeof(*part));

        /* Extract Content-Disposition parameters */
        extract_param(header_str, "name", part->name, sizeof(part->name));
        extract_param(header_str, "filename", part->filename, sizeof(part->filename));

        /* Extract Content-Type if present */
        const char *ct = strstr(header_str, "Content-Type:");
        if (!ct) ct = strstr(header_str, "content-type:");
        if (ct) {
            ct += 13; /* strlen("Content-Type:") */
            while (*ct == ' ') ct++;
            size_t i = 0;
            while (*ct && *ct != '\r' && *ct != '\n' && i < sizeof(part->content_type) - 1) {
                part->content_type[i++] = *ct++;
            }
            part->content_type[i] = '\0';
        }

        /* Part data starts after header_end + \r\n\r\n */
        const unsigned char *data_start = header_end + 4;

        /* Find next boundary */
        const unsigned char *next_bnd = mem_find(data_start, (size_t)(end - data_start),
                                                 bnd, boundary_len);
        if (!next_bnd) {
            /* No more boundaries -- use remaining data */
            part->data = data_start;
            part->data_len = (size_t)(end - data_start);
            count++;
            break;
        }

        /* Data ends before the \r\n before the boundary */
        part->data = data_start;
        part->data_len = (size_t)(next_bnd - data_start);
        /* Strip trailing \r\n before boundary */
        if (part->data_len >= 2 &&
            part->data[part->data_len - 2] == '\r' &&
            part->data[part->data_len - 1] == '\n') {
            part->data_len -= 2;
        }

        count++;

        /* Advance past boundary */
        pos = next_bnd + boundary_len;
    }

    return count;
}

const MultipartPart *multipart_find(const MultipartPart *parts, int count, const char *name) {
    for (int i = 0; i < count; i++) {
        if (strcmp(parts[i].name, name) == 0) {
            return &parts[i];
        }
    }
    return NULL;
}
