#ifndef QSP_ARENA_H
#define QSP_ARENA_H

#include <stddef.h>

typedef struct {
  double *beg;
  double *end;
} qsp_arena;

static inline qsp_arena qsp_arena_make(double *buf, int count) {
  qsp_arena a;
  if (!buf || count <= 0) {
    a.beg = NULL;
    a.end = NULL;
  } else {
    a.beg = buf;
    a.end = buf + count;
  }
  return a;
}

static inline double *qsp_arena_take(qsp_arena *a, int count) {
  if (!a || count < 0 || a->end - a->beg < count)
    return NULL;
  double *p = a->beg;
  a->beg += count;
  return p;
}

#endif /* QSP_ARENA_H */
