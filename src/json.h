/*
 * JSON Utilities
 * Buffer-based JSON writer and minimal JSON reader.
 * No external dependencies.
 */

#ifndef LOCAL_AI_JSON_H
#define LOCAL_AI_JSON_H

#include <stddef.h>
#include <stdint.h>

/* ---- JSON Writer ---- */

#define JW_MAX_ARRAY_DEPTH 8

/* Buffer-based JSON writer state */
typedef struct {
    char *buf;          /* Output buffer (caller-owned) */
    size_t capacity;    /* Buffer capacity */
    size_t len;         /* Current write position */
    int overflow;       /* Set if buffer was exceeded */
    int first_field;    /* Track comma insertion for objects */
    int array_depth;
    int array_first[JW_MAX_ARRAY_DEPTH]; /* Track comma insertion per array level */
} JsonWriter;

/* Initialize writer to use the given buffer */
void jw_init(JsonWriter *w, char *buf, size_t capacity);

/* Object */
void jw_object_start(JsonWriter *w);
void jw_object_end(JsonWriter *w);

/* Array */
void jw_array_start(JsonWriter *w);
void jw_array_end(JsonWriter *w);

/* Separators (auto-handled by field_ helpers) */
void jw_obj_sep(JsonWriter *w);
void jw_array_sep(JsonWriter *w);

/* Raw values */
void jw_string(JsonWriter *w, const char *value);
void jw_int(JsonWriter *w, int64_t value);
void jw_double(JsonWriter *w, double value, int precision);
void jw_bool(JsonWriter *w, int value);
void jw_null(JsonWriter *w);

/* Field helpers (key + value, auto comma) */
void jw_field_string(JsonWriter *w, const char *key, const char *value);
void jw_field_int(JsonWriter *w, const char *key, int64_t value);
void jw_field_double(JsonWriter *w, const char *key, double value, int precision);
void jw_field_bool(JsonWriter *w, const char *key, int value);
void jw_field_null(JsonWriter *w, const char *key);

/* Start a named array field: "key":[ */
void jw_field_array_start(JsonWriter *w, const char *key);

/* Start a named object field: "key":{ */
void jw_field_object_start(JsonWriter *w, const char *key);

/* Reset first-field tracking (call when starting fields of a new object) */
void jw_reset_object(JsonWriter *w);

/* Get the null-terminated result string length */
size_t jw_length(const JsonWriter *w);

#endif /* LOCAL_AI_JSON_H */
