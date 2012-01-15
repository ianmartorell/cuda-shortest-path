#pragma once

#include <string>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include "io.cuh"
#include "help.h"
#include "timer.cuh"

extern int *g; 
extern int *dev_g;

extern int *gSize; 
extern int *dev_gSize;

extern int *gIndex;
extern int *dev_gIndex;

extern int g_size;
extern int edge_num;
extern int num_cols_per_row;

extern int *parent;

void test_ell_bellman(int source, int target, string filename, bool printPath);
int ell_bellman(int source, int target, int size);
__global__ void relax(int *distD, int *g, int *gIndex, int *distS, int size, int num_cols_per_row, bool *change, int *parent);
