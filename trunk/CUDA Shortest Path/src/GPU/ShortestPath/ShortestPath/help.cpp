#include "help.h"
using namespace std;

void help()
{
    cout << "useage: ShortestPath.exe [method] [filename] [source] [target] [printPath]" << endl << endl;
    
    cout << "e.g. : ShortestPath.exe 3 USA-road-d.NY.gr" << endl << endl;
    
    cout << "method: shorest path method (method index or name is either acceptable)" << endl;
    cout << "0) all : All methods will be tested" << endl;
    cout << "1) cuda-dijkstra: CUDA Dijkstra" << endl;
    cout << "2) cuda-bellman: CUDA Bellman Ford" << endl;
    cout << "3) cuda-delta-stepping: CUDA Delta Stepping" << endl;
    cout << "4) cuda-csr-bellman-scalar: CUDA CSR Bellman (Scalar)" << endl;
    cout << "5) cuda-csr-bellman-vector: CUDA CSR Bellman (Vector)" << endl;
    cout << "6) cuda-ell-bellman: CUDA ELL Bellman" << endl << endl;

    cout << "filename: Graph Filenames." << endl << endl;
    
    cout << "source: source node. if not specified, default is 1." << endl << endl;
    
    cout << "target: target node. if not specified, default is the max index node of graph." << endl << endl; 

    cout << "printPath: 0: not print, 1: print. if not specified, default is 0." << endl << endl;
}

