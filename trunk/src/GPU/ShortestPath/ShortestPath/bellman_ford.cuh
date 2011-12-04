#pragma once

#include <string>
#include <iostream>
#include "io.cuh"
#include "help.h"
#include "lock.cuh"
#include "timer.cuh"
using namespace std;

struct Queue
{
    int head, tail;
};

extern int *g; 
extern int *dev_g;

extern int *gSize; 
extern int *dev_gSize;

extern int *gIndex;
extern int *dev_gIndex;

extern int g_size;
extern int edge_num;

extern int *parent;

void test_bellman_ford(int source, int target, string filename, bool printPath);
int bellman_ford(int source, int target);
__global__ void relax(int *dist, int *g, int *gIndex, int *q, Queue *qInfo, bool *canUse, int index, int beg, int size, int qSingleSize, Lock *lock, int *parent);

