#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "io.cuh"
using namespace std;

struct Graph
{
	int node, weight;

    Graph(){}
    Graph(int inNode, int w):node(inNode), weight(w){}
};

vector<list<Graph> > init_g;

int *g; 
int *dev_g;

int *gSize; 
int *dev_gSize;

int *gIndex;
int *dev_gIndex;

int *gRow;
int *dev_gRow;

int g_size, edge_num;
int num_cols_per_row;

int *parent;

bool cmp(const Graph &lhs, const Graph &rhs)
{
    return lhs.node < rhs.node;
}

//init_g[u] strore v in u->v
void ReadGraph(string filename)
{
    ifstream fin(filename.c_str());

    int n, m;
    int inNode, outNode, weight;
    char op;
    string line;

    while(fin >> op)
    {
        if (op == 'c')
        {
            getline(fin, line);
        }
        else if (op == 'p')
        {
            string type;
            fin >> type >> n >> m;
            init_g.resize(n);
        }
        else
        {
            fin >> inNode >> outNode >> weight;

            Graph newNode(inNode - 1, weight);

            init_g[outNode - 1].push_back(newNode);
        }
    }

    cout << "Converting to CSR Graph..." << endl;

    Convert2CSRGraph();

    fin.close();
}

//init_g[v] store u in u->v
void ReadCSRGraph(string filename)
{
    ifstream fin(filename.c_str());

    int n, m;
    int inNode, outNode, weight;
    char op;
    string line;

    while(fin >> op)
    {
        if (op == 'c')
        {
            getline(fin, line);
        }
        else if (op == 'p')
        {
            string type;
            fin >> type >> n >> m;
            init_g.resize(n);
        }
        else
        {
            fin >> inNode >> outNode >> weight;

            Graph newNode(outNode - 1, weight);

            init_g[inNode - 1].push_back(newNode);
        }
    }

    cout << "Converting to CSR Graph..." << endl;

    Convert2CSRGraph();

    fin.close();
}

void ReadELLGraph(string filename)
{
    ifstream fin(filename.c_str());

    int n, m;
    int inNode, outNode, weight;
    char op;
    string line;

    while(fin >> op)
    {
        if (op == 'c')
        {
            getline(fin, line);
        }
        else if (op == 'p')
        {
            string type;
            fin >> type >> n >> m;
            init_g.resize(n);
        }
        else
        {
            fin >> inNode >> outNode >> weight;

            Graph newNode(outNode - 1, weight);

            init_g[inNode - 1].push_back(newNode);
        }
    }

    cout << "Converting to ELL Graph..." << endl;

    Convert2ELLGraph();

    fin.close();
}

void Convert2ELLGraph()
{
    g_size = init_g.size();

    num_cols_per_row = INT_MIN; // determine the max num in row of matrix

    for(int i = 0; i < init_g.size(); i++)
        num_cols_per_row = max(num_cols_per_row, (int)init_g[i].size());

    int edge_num = num_cols_per_row * g_size;

    g = (int *)malloc(edge_num * sizeof(int));
    gIndex = (int *)malloc(edge_num * sizeof(int));
    memset(gIndex, -1, edge_num * sizeof(int));
        
    cudaMalloc(&dev_g, edge_num * sizeof(int));
    cudaMalloc(&dev_gIndex, edge_num * sizeof(int));

    for(int i = 0; i < init_g.size(); i++)
    {
        list<Graph>::iterator edgeIter = init_g[i].begin();
        int j = 0;
        for( ; edgeIter != init_g[i].end(); edgeIter++, j++)
        {
            g[j * g_size + i] = edgeIter->weight;
            gIndex[j * g_size + i] = edgeIter->node;
        }
    }

    cudaMemcpy(dev_g, g, edge_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gIndex, gIndex, edge_num * sizeof(int), cudaMemcpyHostToDevice);

    parent = (int *)malloc(g_size * sizeof(int));

    init_g.clear();
}

void Convert2CSRGraph()
{
    g_size = init_g.size();

    edge_num = 0;

    for(int i = 0; i < init_g.size(); i++)
        edge_num += init_g[i].size();

    g = (int *)malloc(edge_num * sizeof(int));
    gIndex = (int *)malloc(edge_num * sizeof(int));
    gSize = (int *)malloc((g_size + 1) * sizeof(int));

    cudaMalloc(&dev_g, edge_num * sizeof(int));
    cudaMalloc(&dev_gIndex, edge_num * sizeof(int));
    cudaMalloc(&dev_gSize, (g_size + 1) * sizeof(int));
    
    int g_index = 0;
    gSize[0] = 0;

    for(int i = 0; i < init_g.size(); i++)
    {
        init_g[i].sort(cmp); //sort the nodes may be more effective when relax

        list<Graph>::iterator edgeIter = init_g[i].begin();

        for( ; edgeIter != init_g[i].end(); edgeIter++, g_index++)
        {
            g[g_index] = edgeIter->weight;
            gIndex[g_index] = edgeIter->node;
        }

        gSize[i + 1] = gSize[i] + init_g[i].size();
        
        init_g[i].clear();
    }

    cudaMemcpy(dev_g, g, edge_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gIndex, gIndex, edge_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gSize, gSize, (g_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    parent = (int *)malloc(g_size * sizeof(int));
    
    init_g.clear();
}

void PrintResult(string method, int source, int target, int dist, double time, bool printPath)
{
    cout << "Method: " << method << endl;
    if (dist != INT_MAX)
        cout << "Shortest distance from " << source + 1 << " to " << target + 1 << " is: " << dist << endl;
    else
        cout << "There is no way from " << source + 1 << " to " << target + 1 << endl;
    cout << "Time: " << time << "secs" << endl << endl;

    if (printPath)
    {
        cout << "Shortest Path from " << source + 1 << " to " << target + 1 << " is:" << endl;
        PrintPath(parent, target, target);
        cout << endl;
    }
}

void PrintPath(int *parent, int index, int target)
{
    if (parent[index] != -1)
        PrintPath(parent, parent[index], target);

    if (index != target)
        cout << index + 1 << "-->";
    else
        cout << index + 1 << endl;
}
