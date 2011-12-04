// CUDA 2011 Programming Contest
// Author: Chen Kai
// Update : clean code, make some comments
// Update : 2011/10/13 finish paper, add PrintPath
// Update : 2011/10/12 fix lock, add comment, refactoring
// Update : 2011/10/6   For Steve

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <string>
#include "dijkstra.cuh"
#include "bellman_ford.cuh"
#include "delta_stepping.cuh"
#include "csr_bellman.cuh"
#include "ell_bellman.cuh"
using namespace std;

int main(int argc, char *argv[])
{
    string filename; // graph name
    string method; // which CUDA Algorithm
    int source = 0; // 0 is source node
    int target = -1; // -1 is largest index node
    bool printPath = false;
    
    if (argc >= 2)
    {
        if (strcmp(argv[1], "-h") == 0)
        {
            help();
            return 0;
        }
        else
            method = argv[1];
    }
    if (argc >= 3)
        filename = argv[2];
    if (argc >= 4)
        source = atoi(argv[3]) - 1; // graph is indexed from 0
    if (argc >= 5)
        target = atoi(argv[4]) - 1;
    if (argc >= 6)
        printPath = atoi(argv[5]);

    if (method == "all" || method == "0")
    {
        test_dijkstra(source, target, filename, printPath);

        test_bellman_ford(source, target, filename, printPath);

        test_delta_stepping_dijkstra(source, target, filename, printPath);

        test_csr_bellman(source, target, filename, printPath, true);

        test_csr_bellman(source, target, filename, printPath, false);

        test_ell_bellman(source, target, filename, printPath);
    }
    else if (method == "cuda-dijkstra" || method == "1")
        test_dijkstra(source, target, filename, printPath);
    else if (method == "cuda-bellman" || method == "2")
        test_bellman_ford(source, target, filename, printPath);
    else if (method == "cuda-delta-stepping" || method == "3")
        test_delta_stepping_dijkstra(source, target, filename, printPath);
    else if (method == "cuda-csr-bellman-scalar" || method == "4")
        test_csr_bellman(source, target, filename, printPath, true);
    else if (method == "cuda-csr-bellman-vector" || method == "5")
        test_csr_bellman(source, target, filename, printPath, false);
    else if (method == "cuda-ell-bellman" || method == "6")
        test_ell_bellman(source, target, filename, printPath);

    return 0;
}



