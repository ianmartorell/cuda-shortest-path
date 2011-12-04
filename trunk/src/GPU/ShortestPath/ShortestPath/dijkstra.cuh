#pragma once

#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <iostream>
#include "io.cuh"
#include "help.h"
#include "timer.cuh"
using namespace std;
using namespace thrust;

extern int *g; 
extern int *dev_g;

extern int *gSize; 
extern int *dev_gSize;

extern int *gIndex;
extern int *dev_gIndex;

extern int g_size;
extern int edge_num;

extern int *parent;

void test_dijkstra(int source, int target, string filename, bool printPath);
int dijkstra(int source, int target);
tuple<int, int> ExtractMin(device_vector<int> &dist);
__global__ void relax(int *dist, bool *canUse, int minDist, int *g, int *gIndex, int index, int beg, int size, int *parent);
