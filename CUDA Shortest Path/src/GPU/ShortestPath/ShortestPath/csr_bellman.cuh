#pragma once

#include <string>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <iostream>
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

void test_csr_bellman(int source, int target, string filename, bool printPath, bool isScalar);
int csr_bellman_scalar(int source, int target, int size);
int csr_bellman_vector(int source, int target, int size);

template<int VectorsPerBlock, int ThreadsPerVector>
__global__ void relax_vector(int *distD, int *g, int *gIndex, int *gSize, int *distS, int size, bool *change, int *parent);
__global__ void relax_scalar(int *distD, int *g, int *gIndex, int *gSize, int *distS, int size, bool *change, int *parent);
