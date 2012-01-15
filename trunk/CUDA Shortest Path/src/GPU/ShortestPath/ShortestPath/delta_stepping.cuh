#pragma once

#include <string>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <algorithm>
#include <iostream>
#include "lock.cuh"
#include "io.cuh"
#include "help.h"
#include "timer.cuh"
using namespace std;

extern int *g; 
extern int *dev_g;

extern int *gSize; 
extern int *dev_gSize;

extern int *gIndex;
extern int *dev_gIndex;

extern int g_size;
extern int edge_num;

extern int *parent;

void test_delta_stepping_dijkstra(int source, int target, string filename, bool printPath);
int delta_stepping_dijkstra(int source, int target, int size, int delta);
__global__ void relax(int *BIndex, int *dist, int *g, int *gIndex, int *gSize, int *req, int *reqSize, int sum, Lock *lock, int delta, int type, int *parent);
__global__ void Add2Req(int *BIndex, int *S, int *ReqIndex, int *ds_gSize, int *req, int *reqSize, int BCurIndex, int size);
__global__ void Add2Req(int *S, int *ReqIndex, int *ds_gSize, int *req, int *reqSize, int size);

int MeanDelta();
int MidDelta();
int EdgeDegreeDelta();
