#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bellman_ford.cuh"
using namespace std;

__global__ void LockInit_Bellman(Lock *lock, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (tid < size)
    {
        lock[tid].mutex = 0;
        tid += offset;
    }
}

void test_bellman_ford(int source, int target, string filename, bool printPath)
{
    cout << "Reading Graph..." << endl;

    ReadGraph(filename);

    if (target == -1)
        target = g_size - 1;

    cout << "Doing ShortestPath Algorithm..." << endl;

    CUDATimer timer;

    int dist = bellman_ford(source, target);

    string method = "CUDA Bellman Ford";

    PrintResult(method, source, target, dist, timer.seconds_elapsed(), printPath);

    free(g);
    free(gSize);
    free(gIndex);
    free(parent);
    cudaFree(dev_g);
    cudaFree(dev_gSize);
    cudaFree(dev_gIndex);
}

int bellman_ford(int source, int target)
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
    
    int *dist;
    HANDLE_ERROR(cudaHostAlloc(&dist, sizeof(int) * g_size, cudaHostAllocMapped));
    
    for(int i = 0; i < g_size; i++)
        dist[i] = INT_MAX;
    dist[source] = 0;

    int *dev_dist;
    
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_dist, dist, 0));

    int *dev_parent;
    HANDLE_ERROR(cudaMalloc(&dev_parent, sizeof(int) * g_size));
    cudaMemset(dev_parent, -1, sizeof(int) * g_size);

    bool *dev_canUse;
    HANDLE_ERROR(cudaMalloc(&dev_canUse, sizeof(bool) * g_size));
    HANDLE_ERROR(cudaMemset(dev_canUse, true, sizeof(bool) * g_size));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threadsPerBlock = prop.maxThreadsDim[0];
    const int blocksPerGrid = std::min(prop.maxGridSize[0], (g_size + threadsPerBlock - 1) / threadsPerBlock);

    int qSize = blocksPerGrid; // the first axis size of q
    int qSingleSize = threadsPerBlock;  // the second axis size of q
    
    int *q; // q is 2D matrix, I map it to 1D matrix 
    HANDLE_ERROR(cudaHostAlloc(&q, sizeof(int) * (qSize * qSingleSize), cudaHostAllocMapped));
    Queue *qInfo;
    HANDLE_ERROR(cudaHostAlloc(&qInfo, sizeof(Queue) * qSize, cudaHostAllocMapped));

    for(int i = 0; i < qSize; i++)
        qInfo[i].head = qInfo[i].tail = 0;

    int qSourceIdx = (source / qSingleSize) * qSingleSize;
    int qIndex = qSourceIdx;
    q[qSourceIdx] = source;
    qInfo[qSourceIdx].head = 0;
    qInfo[qSourceIdx].tail = 1;

    int *dev_q;
    cudaHostGetDevicePointer(&dev_q, q, 0);
    Queue *dev_qInfo;
    cudaHostGetDevicePointer(&dev_qInfo, qInfo, 0);

    Lock *dev_lock;
    HANDLE_ERROR(cudaMalloc(&dev_lock, sizeof(Lock) * g_size));

    LockInit_Bellman<<<blocksPerGrid, threadsPerBlock>>>(dev_lock, g_size);

    while(true)
    {
        int old_qIndex = qIndex;
        int index = -1;

        while (true)
        {
            if (qInfo[qIndex].head == qInfo[qIndex].tail)
            {
                qIndex = (qIndex + 1) % qSize;
                if (qIndex == old_qIndex)
                    break;
            }
            else
            {
                int size = qInfo[qIndex].tail - qInfo[qIndex].head;
                if (size < 0)
                    size = qSingleSize + size;
                    
                int beforeThreadsSum = qIndex * qSingleSize;

                //取出q的首元素
                index = q[qInfo[qIndex].head + beforeThreadsSum];
                qInfo[qIndex].head = (qInfo[qIndex].head + 1) % qSingleSize;
        
                break;
            }            
        }

        if (index == -1)
            break;

        relax<<<blocksPerGrid, threadsPerBlock>>>
            (dev_dist, dev_g, dev_gIndex, dev_q, dev_qInfo, dev_canUse, index, gSize[index], gSize[index + 1] - gSize[index], qSingleSize, dev_lock, dev_parent);

        HANDLE_ERROR(cudaThreadSynchronize());
    }

    cudaMemcpy(parent, dev_parent, sizeof(int) * g_size, cudaMemcpyDeviceToHost);

    int ret = dist[target];

    cudaFreeHost(dist);
    cudaFree(dev_canUse);
    cudaFreeHost(q);
    cudaFreeHost(qInfo);
    cudaFree(dev_lock);
    cudaFree(dev_parent);
    
    return ret;
}

__global__ void relax(int *dist, int *g, int *gIndex, int *q, Queue *qInfo, bool *canUse, int index, int beg, int size, int qSingleSize, Lock *lock, int *parent)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    if (tid == 0)
        canUse[index] = true;

    while (tid < size)
    {
        int gIdx = tid + beg;
        int newDist = dist[index] + g[gIdx];
        int node = gIndex[gIdx];

        if (newDist < dist[node])
        {
            dist[node] = newDist;
            parent[node] = index;

            if (canUse[node])
            {
                canUse[node] = false;

                int lockIdx = node / qSingleSize; // find the corresponding index of q
                int beforeThreadsSum = lockIdx * qSingleSize;

                for (int i = 0; i < 32; i++)
                    if ((tid % 32) == i)
                    {
                        lock[lockIdx].lock(); 

                        //SLF优化
                        if (qInfo[lockIdx].tail == qInfo[lockIdx].head || newDist < dist[q[qInfo[lockIdx].head + beforeThreadsSum]])
                        {
                            qInfo[lockIdx].head = (qInfo[lockIdx].head - 1 + qSingleSize) % qSingleSize;
                            q[qInfo[lockIdx].head + beforeThreadsSum] = gIndex[gIdx];
                        }
                        else
                        {
                            q[qInfo[lockIdx].tail + beforeThreadsSum] = gIndex[gIdx];
                            qInfo[lockIdx].tail = (qInfo[lockIdx].tail + 1) % qSingleSize;
                        }
                     
                        lock[lockIdx].unlock();         
                    }
            }
        }
        tid += offset;
    }
}


