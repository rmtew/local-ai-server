/*
 * json_reader.h - Minimal JSON object reader
 *
 * Extracts string and number fields from flat JSON objects.
 * Not a general-purpose parser -- handles the subset needed for
 * OpenAI-compatible API request bodies.
 */

#ifndef LOCAL_AI_JSON_READER_H
#define LOCAL_AI_JSON_READER_H

#include <stddef.h>

/* Extract a string value for the given key from a JSON object.
 * Writes unescaped value into out_buf (null-terminated).
 * Returns length of value, or -1 if key not found. */
int jr_get_string(const char *json, size_t json_len,
                  const char *key, char *out_buf, size_t buf_size);

/* Extract a number value for the given key from a JSON object.
 * Returns 0 on success with value in *out, or -1 if key not found. */
int jr_get_double(const char *json, size_t json_len,
                  const char *key, double *out);

/* Extract a boolean value for the given key from a JSON object.
 * Sets *out to 1 for true, 0 for false.
 * Returns 0 on success, or -1 if key not found or not a boolean. */
int jr_get_bool(const char *json, size_t json_len,
                const char *key, int *out);

#endif /* LOCAL_AI_JSON_READER_H */
