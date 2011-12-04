#pragma once

#include <list>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "error.h"
using namespace std;

extern int *g; 
extern int *dev_g;

extern int *gSize; 
extern int *dev_gSize;

extern int *gIndex;
extern int *dev_gIndex;

extern int *gRow;
extern int *dev_gRow;

extern int g_size;
extern int edge_num;
extern int num_cols_per_row;

extern int *parent;

void ReadGraph(string filename);
void ReadCSRGraph(string filename);
void Convert2CSRGraph();
void Convert2ELLGraph();
void ReadELLGraph(string filename);
void PrintPath(int *parent, int index, int target);
void PrintResult(string method, int source, int target, int dist, double time, bool printPath);
