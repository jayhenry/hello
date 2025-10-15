// point.c
#include "point.h"

// 一个操作Point的函数
void add_points(Point* a, const Point* b, const Point* c) {
    a->x = b->x + c->x;
    a->y = b->y + c->y;
}