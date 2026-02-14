/*
 * json_reader.c - Minimal JSON object reader
 *
 * Handles flat JSON objects with string and number values.
 * Does not support nested objects, arrays, or unicode escapes beyond \uXXXX.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "json_reader.h"

#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* Skip whitespace, return pointer to next non-space char */
static const char *skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Find a key in a JSON object. Returns pointer to the value start (after colon),
 * or NULL if not found. */
static const char *find_key(const char *json, size_t json_len, const char *key) {
    const char *end = json + json_len;
    size_t key_len = strlen(key);

    const char *p = json;
    while (p < end) {
        /* Find next quote */
        const char *q = memchr(p, '"', (size_t)(end - p));
        if (!q) return NULL;

        const char *key_start = q + 1;
        /* Find closing quote for this key (no escape handling for keys) */
        const char *key_end = memchr(key_start, '"', (size_t)(end - key_start));
        if (!key_end) return NULL;

        size_t this_key_len = (size_t)(key_end - key_start);

        /* Check if this key matches */
        if (this_key_len == key_len && memcmp(key_start, key, key_len) == 0) {
            /* Skip to colon */
            const char *colon = key_end + 1;
            colon = skip_ws(colon, end);
            if (colon < end && *colon == ':') {
                return skip_ws(colon + 1, end);
            }
        }

        /* Skip past the value to continue searching */
        p = key_end + 1;
        p = skip_ws(p, end);
        if (p < end && *p == ':') {
            p = skip_ws(p + 1, end);
            /* Skip value */
            if (p < end && *p == '"') {
                /* String value -- skip to closing quote, handling escapes */
                p++;
                while (p < end) {
                    if (*p == '\\') { p += 2; continue; }
                    if (*p == '"') { p++; break; }
                    p++;
                }
            } else {
                /* Number/bool/null -- skip to next delimiter */
                while (p < end && *p != ',' && *p != '}') p++;
            }
        }
    }
    return NULL;
}

int jr_get_string(const char *json, size_t json_len,
                  const char *key, char *out_buf, size_t buf_size) {
    const char *end = json + json_len;
    const char *val = find_key(json, json_len, key);
    if (!val || val >= end || *val != '"') return -1;

    val++; /* skip opening quote */
    size_t out_len = 0;

    while (val < end && *val != '"') {
        char c;
        if (*val == '\\') {
            val++;
            if (val >= end) break;
            switch (*val) {
            case '"':  c = '"';  break;
            case '\\': c = '\\'; break;
            case '/':  c = '/';  break;
            case 'n':  c = '\n'; break;
            case 'r':  c = '\r'; break;
            case 't':  c = '\t'; break;
            case 'b':  c = '\b'; break;
            case 'f':  c = '\f'; break;
            default:   c = *val; break; /* including \uXXXX passthrough */
            }
        } else {
            c = *val;
        }
        if (out_len + 1 < buf_size) {
            out_buf[out_len++] = c;
        }
        val++;
    }

    if (out_len < buf_size) out_buf[out_len] = '\0';
    else if (buf_size > 0) out_buf[buf_size - 1] = '\0';

    return (int)out_len;
}

int jr_get_double(const char *json, size_t json_len,
                  const char *key, double *out) {
    const char *end = json + json_len;
    const char *val = find_key(json, json_len, key);
    if (!val || val >= end) return -1;

    /* Must start with digit, minus, or dot */
    if (!isdigit((unsigned char)*val) && *val != '-' && *val != '.') return -1;

    char buf[64];
    size_t i = 0;
    while (val < end && i < sizeof(buf) - 1 &&
           (isdigit((unsigned char)*val) || *val == '.' || *val == '-' || *val == '+' ||
            *val == 'e' || *val == 'E')) {
        buf[i++] = *val++;
    }
    buf[i] = '\0';

    char *endp;
    *out = strtod(buf, &endp);
    return (endp > buf) ? 0 : -1;
}
