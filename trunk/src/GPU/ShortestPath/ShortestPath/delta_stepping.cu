#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "delta_stepping.cuh"
using namespace std;
using namespace thrust;

// determine whether BIndex[i] == BCurIndex
struct BIndex_functor
{
    const int i;

    BIndex_functor(int _i):i(_i){}

    __device__ int operator()(const int &b)
    {
        return (b == i ? 1 : 0);
    }
};

// Initialize lock to 0
__global__ void LockInit_DS(Lock *lock, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (tid < size)
    {
        lock[tid].mutex = 0;
        tid += offset;
    }
}

void test_delta_stepping_dijkstra(int source, int target, string filename, bool printPath)
{
    cout << "Reading Graph..." << endl;

    ReadGraph(filename);

    if (target == -1)
        target = g_size - 1;

    int delta;

    //delta = MeanDelta();
    //delta = MidDelta();
    delta = EdgeDegreeDelta(); //EdgeDegreeDelta is more effective

    cout << "Doing ShortestPath Algorithm..." << endl;

    CUDATimer timer;

    int dist = delta_stepping_dijkstra(source, target, g_size, delta);

    string method = "CUDA Delta Stepping";
    
    PrintResult(method, source, target, dist, timer.seconds_elapsed(), printPath);

    free(g);
    free(gSize);
    free(gIndex);
    free(parent);
    cudaFree(dev_g);
    cudaFree(dev_gSize);
    cudaFree(dev_gIndex);
}

int delta_stepping_dijkstra(int source, int target, int size, int delta)
{
    device_vector<int> dist(size, INT_MAX);
    dist[source] = 0;
    int *dev_dist = raw_pointer_cast(&dist[0]);

    device_vector<int> dev_parent(size, -1);
    int *dev_parent_ptr = raw_pointer_cast(&dev_parent[0]);

    device_vector<int> BIndex(size, -1); // -1 belongs to no Bucket
    BIndex[source] = 0; // initial bucket is 0
    int *dev_BIndex = raw_pointer_cast(&BIndex[0]);

    device_vector<int> S(size, 0);
    int *dev_S = raw_pointer_cast(&S[0]);

    device_vector<int> ReqIndex(size);
    int *dev_ReqIndex = raw_pointer_cast(&ReqIndex[0]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threadsPerBlock = prop.maxThreadsDim[0];
    const int blocksPerGrid = std::min(prop.maxGridSize[0], (size + threadsPerBlock - 1) / threadsPerBlock);

    Lock *dev_lock;
    HANDLE_ERROR(cudaMalloc(&dev_lock, sizeof(Lock) * size));
    LockInit_DS<<<blocksPerGrid, threadsPerBlock>>>(dev_lock, size);
    
    device_vector<int> req(size);
    int *dev_req = raw_pointer_cast(&req[0]);

    device_vector<int> reqSize(size);
    int *dev_reqSize = raw_pointer_cast(&reqSize[0]);

    int BCurIndex = 0;
    int sum = 0;

    while (true)
    {
        //find the biggest value in BIndex£¬if biggest < BCurIndex then bucket is empty, break
        int biggest = reduce(BIndex.begin(), BIndex.end(), -1, thrust::maximum<int>());

        if (biggest < BCurIndex)
            break;

        while(true)
        {
            // if BIndex[i] == BCurIndex then return 1, and then use inclusive_scan to fill ReqIndex
            thrust::transform_inclusive_scan(BIndex.begin(), BIndex.end(), ReqIndex.begin(), BIndex_functor(BCurIndex), thrust::plus<int>());
            
            int reqSize_num = ReqIndex[size - 1];

            if (reqSize_num == 0)
                break;

            Add2Req<<<blocksPerGrid, threadsPerBlock>>>(dev_BIndex, dev_S, dev_ReqIndex, dev_gSize, dev_req, dev_reqSize, BCurIndex, size);

            device_vector<int>::iterator reqSize_end = reqSize.begin() + reqSize_num;
            sum = reduce(reqSize.begin(), reqSize_end, 0, thrust::plus<int>()); // the sum of all edges need to be relaxed

            relax<<<blocksPerGrid, threadsPerBlock>>>(dev_BIndex, dev_dist, dev_g, dev_gIndex, dev_gSize, dev_req, dev_reqSize, sum, dev_lock, delta, 0, dev_parent_ptr);
        }

        BCurIndex++;

        thrust::inclusive_scan(S.begin(), S.end(), ReqIndex.begin());

        int reqSize_num = ReqIndex[size - 1];

        if (reqSize_num == 0)
            continue;

        Add2Req<<<blocksPerGrid, threadsPerBlock>>>(dev_S, dev_ReqIndex, dev_gSize, dev_req, dev_reqSize, size);

        device_vector<int>::iterator reqSize_end = reqSize.begin() + reqSize_num;
        sum = reduce(reqSize.begin(), reqSize_end, 0, thrust::plus<int>());

        relax<<<blocksPerGrid, threadsPerBlock>>>(dev_BIndex, dev_dist, dev_g, dev_gIndex, dev_gSize, dev_req, dev_reqSize, sum, dev_lock, delta, 1, dev_parent_ptr);
    }

    cudaMemcpy(parent, dev_parent_ptr, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaFree(dev_lock);

    int ret = dist[target];

    return ret;    
}

// type = 0 : light relax, type = 1 : heavy relax
__global__ void relax(int *BIndex, int *dist, int *g, int *gIndex, int *gSize, int *req, int *reqSize, int sum, Lock *lock, int delta, int type, int *parent) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (tid < sum)
    {
        int size = tid;
        int req_index = 0;

        while (size >= reqSize[req_index])
        {
            size -= reqSize[req_index];
            req_index++;
        }

        int reqNode = req[req_index];
        int g_index = gSize[reqNode] + size;
        int d = dist[reqNode];        
        int w = g[g_index];

        // determine the edge belongs to light or heavy? type == 0: light relax, type == 1: heavy relax 
        if ((type == 0 && w > delta) || (type == 1 && w <= delta))
        {
            tid += offset;
            continue;
        }

        int node = gIndex[g_index];

        //use loops to avoid the thread in the same warp to compete the same lock        
        for(int i = 0; i < 32; i++)
            if ((tid % 32) == i)
            {
                lock[node].lock();

                if (d + w < dist[node])
                {
                    BIndex[node] = (d + w) / delta; //update bucket
                    dist[node] = d + w;
                    parent[node] = reqNode;
                }

                lock[node].unlock();
            }
            
        tid += offset;
    }
}

__global__ void Add2Req(int *BIndex, int *S, int *ReqIndex, int *gSize, int *req, int *reqSize, int BCurIndex, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (tid < size)
    {
        if (BIndex[tid] == BCurIndex)
        {
            S[tid] = 1;
            BIndex[tid] = -1;

            int idx = ReqIndex[tid] - 1; // get the req index
            req[idx] = tid;
            reqSize[idx] = gSize[tid + 1] - gSize[tid]; // get how many edge of node tid link
        }

        tid += offset;
    }
}

__global__ void Add2Req(int *S, int *ReqIndex, int *gSize, int *req, int *reqSize, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (tid < size)
    {
        if (S[tid] == 1)
        {
            S[tid] = 0;
            int idx = ReqIndex[tid] - 1; 
            req[idx] = tid;
            reqSize[idx] = gSize[tid + 1] - gSize[tid];                   
        }

        tid += offset;
    }
}

int EdgeDegreeDelta()
{
    int maxEdge = INT_MIN;

    for(int i = 0; i < edge_num; i++)
        maxEdge = std::max(maxEdge, g[i]);

    int maxDegree = INT_MIN;

    for(int i = 0; i < g_size; i++)
        maxDegree = std::max(maxDegree, gSize[i + 1] - gSize[i]);

    return maxEdge / maxDegree;
}

int MeanDelta()
{
    double sumDist = 0;

    for(int i = 0; i < edge_num; i++)
        sumDist += g[i];

    int ret = (int)(sumDist / edge_num);

    return ret;
}

int MidDelta()
{
    std::sort(g, g + g_size);

    int mid = g[g_size / 2];

    return mid;
}

