/*
 * JSON Utilities
 * Buffer-based JSON writer.
 * Standalone, no external dependencies.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "json.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---- JSON Writer ---- */

static void jw_append(JsonWriter *w, const char *str, size_t len) {
    if (w->overflow) return;
    if (w->len + len >= w->capacity) {
        w->overflow = 1;
        return;
    }
    memcpy(w->buf + w->len, str, len);
    w->len += len;
    w->buf[w->len] = '\0';
}

static void jw_append_str(JsonWriter *w, const char *str) {
    jw_append(w, str, strlen(str));
}

static void jw_append_char(JsonWriter *w, char c) {
    jw_append(w, &c, 1);
}

void jw_init(JsonWriter *w, char *buf, size_t capacity) {
    w->buf = buf;
    w->capacity = capacity;
    w->len = 0;
    w->overflow = 0;
    w->first_field = 1;
    w->array_depth = 0;
    if (capacity > 0) {
        buf[0] = '\0';
    }
}

void jw_object_start(JsonWriter *w) {
    jw_append_char(w, '{');
    w->first_field = 1;
}

void jw_object_end(JsonWriter *w) {
    jw_append_char(w, '}');
}

void jw_array_start(JsonWriter *w) {
    jw_append_char(w, '[');
    if (w->array_depth < JW_MAX_ARRAY_DEPTH) {
        w->array_first[w->array_depth] = 1;
        w->array_depth++;
    }
}

void jw_array_end(JsonWriter *w) {
    jw_append_char(w, ']');
    if (w->array_depth > 0) {
        w->array_depth--;
    }
}

void jw_obj_sep(JsonWriter *w) {
    if (!w->first_field) {
        jw_append_char(w, ',');
    }
    w->first_field = 0;
}

void jw_array_sep(JsonWriter *w) {
    if (w->array_depth > 0 && w->array_depth <= JW_MAX_ARRAY_DEPTH) {
        int idx = w->array_depth - 1;
        if (!w->array_first[idx]) {
            jw_append_char(w, ',');
        }
        w->array_first[idx] = 0;
    } else if (w->array_depth > JW_MAX_ARRAY_DEPTH) {
        jw_append_char(w, ',');
    }
}

void jw_reset_object(JsonWriter *w) {
    w->first_field = 1;
}

void jw_string(JsonWriter *w, const char *value) {
    if (!value) {
        jw_append_str(w, "null");
        return;
    }
    jw_append_char(w, '"');
    for (const char *p = value; *p; p++) {
        switch (*p) {
            case '"':  jw_append_str(w, "\\\""); break;
            case '\\': jw_append_str(w, "\\\\"); break;
            case '\b': jw_append_str(w, "\\b"); break;
            case '\f': jw_append_str(w, "\\f"); break;
            case '\n': jw_append_str(w, "\\n"); break;
            case '\r': jw_append_str(w, "\\r"); break;
            case '\t': jw_append_str(w, "\\t"); break;
            default:
                if ((unsigned char)*p < 32) {
                    char esc[8];
                    snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)*p);
                    jw_append_str(w, esc);
                } else {
                    jw_append_char(w, *p);
                }
                break;
        }
    }
    jw_append_char(w, '"');
}

void jw_int(JsonWriter *w, int64_t value) {
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%lld", (long long)value);
    jw_append_str(w, tmp);
}

void jw_double(JsonWriter *w, double value, int precision) {
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "%.*f", precision, value);
    jw_append_str(w, tmp);
}

void jw_bool(JsonWriter *w, int value) {
    jw_append_str(w, value ? "true" : "false");
}

void jw_null(JsonWriter *w) {
    jw_append_str(w, "null");
}

void jw_string_raw(JsonWriter *w, const char *value, size_t len) {
    if (!value || len == 0) {
        jw_append_str(w, "null");
        return;
    }
    jw_append_char(w, '"');
    jw_append(w, value, len);
    jw_append_char(w, '"');
}

void jw_field_string_raw(JsonWriter *w, const char *key, const char *value, size_t len) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_string_raw(w, value, len);
}

void jw_field_string(JsonWriter *w, const char *key, const char *value) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_string(w, value);
}

void jw_field_int(JsonWriter *w, const char *key, int64_t value) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_int(w, value);
}

void jw_field_double(JsonWriter *w, const char *key, double value, int precision) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_double(w, value, precision);
}

void jw_field_bool(JsonWriter *w, const char *key, int value) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_bool(w, value);
}

void jw_field_null(JsonWriter *w, const char *key) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_null(w);
}

void jw_field_array_start(JsonWriter *w, const char *key) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_array_start(w);
}

void jw_field_object_start(JsonWriter *w, const char *key) {
    jw_obj_sep(w);
    jw_string(w, key);
    jw_append_char(w, ':');
    jw_object_start(w);
}

size_t jw_length(const JsonWriter *w) {
    return w->len;
}
