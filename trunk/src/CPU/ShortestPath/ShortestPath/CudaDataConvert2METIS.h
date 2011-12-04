#include "stdafx.h"
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <string>
using namespace std;

string CudaDataConvert2METIS(string filename);
void OutputMETISGraph(string filename);
void ReadRandomGraph(string filename);
void ReadGraph(string filename);