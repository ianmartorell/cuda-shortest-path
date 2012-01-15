#include "stdafx.h"
#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/distributed/delta_stepping_shortest_paths.hpp>
#include <boost/graph/distributed/crauser_et_al_shortest_paths.hpp>
#include <boost/graph/distributed/eager_dijkstra_shortest_paths.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/metis.hpp>
#include <boost/timer.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <ctime>
#include <limits>
#include "CudaDataConvert2METIS.h"
using namespace boost;
using boost::graph::distributed::mpi_process_group;
using boost::graph::distributed::crauser_et_al_shortest_paths;
using boost::graph::distributed::eager_dijkstra_shortest_paths;
using boost::graph::distributed::delta_stepping_shortest_paths;
using namespace std;

void help();
void PrintResult();
void test_crauser_dijkstra(string filename, int source, int target);
void test_delta_stepping_dijkstra(string filename, int source, int target);
void test_sequential_dijkstra(string filename, int source, int target);
void test_bellman_ford(string filename, int source, int target);

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc,argv);

    string filename;
    string method;
    int source = 0;
    int target = -1;
    
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
        source = atoi(argv[3]) - 1;
    if (argc >= 5)
        target = atoi(argv[4]) - 1;

    if (method == "all" || method == "0")
    {
        test_sequential_dijkstra(filename, source, target);
        test_crauser_dijkstra(filename, source, target);
        test_delta_stepping_dijkstra(filename, source, target);
        test_bellman_ford(filename, source, target);
    }
    else if (method == "dijkstra" || method == "1")
        test_sequential_dijkstra(filename, source, target);
    else if (method == "crauser" || method == "2")
        test_crauser_dijkstra(filename, source, target);
    else if (method == "delta-stepping" || method == "3")
        test_delta_stepping_dijkstra(filename, source, target);
    else if (method == "bellman-ford" || method == "4")
        test_bellman_ford(filename, source, target);

    return 0;
}

void help()
{
    cout << "useage: ShortestPath.exe [method] [filename] [source] [target]" << endl << endl;
    cout << "e.g. : ShortestPath.exe sequential USA-road-d.NY.gr" << endl << endl;
    cout << "method: shorest path method (method index or name is either acceptable)" << endl;
    cout << "0) all : All methods will be tested" << endl;
    cout << "1) dijkstra : Dijkstra" << endl;
    cout << "2) crauser : Crauser" << endl;
    cout << "3) delta-stepping : Delta Stepping" << endl;
    cout << "4) bellman-ford : Bellman Ford" << endl << endl;
    cout << "filename: Graph Filenames" << endl << endl;
    cout << "source: source node. if not specified, default is 1" << endl << endl;
    cout << "target: target node. if not specified, default is the max index node of graph" << endl << endl; 
}

void PrintResult(string method, int source, int target, int dist, double time)
{
    cout << "Method: " << method << endl;
    cout << "Shortest distance from " << source + 1 << " to " << target + 1 << " is: " << dist << endl;
    cout << "Time: " << time << "secs" << endl << endl;
}

void test_bellman_ford(string filename, int source, int target)
{
    cout << "Converting To METIS Graph File..." << endl;
    filename = CudaDataConvert2METIS(filename);
    
    typedef property<vertex_distance_t, int> VertexProperty;
    typedef property<edge_weight_t, int> EdgeProperty;
    typedef adjacency_list<listS, vecS, directedS, VertexProperty , EdgeProperty> Graph;

    cout << "Reading Graph..." << endl;
    ifstream fin(filename.c_str());
    graph::metis_reader reader(fin);

    cout << "Constrcting Graph..." << endl;
    Graph g(reader.begin(), reader.end(), reader.weight_begin(), reader.num_vertices());

    vector<int> dist(num_vertices(g), numeric_limits<int>::max());
    dist[source] = 0;

    if (target == -1)
        target = num_vertices(g) - 1;

    cout << "Doing Shortest Alogrithm..." << endl;

    fin.close();

    timer time;

    bellman_ford_shortest_paths(g, num_vertices(g), distance_map(&dist[0]));

    PrintResult("Bellman Ford", source, target, dist[target], time.elapsed());

    g.clear();

    string cmd = "del " + filename;
    std::system(cmd.c_str());
}

void test_crauser_dijkstra(string filename, int source, int target)
{
    cout << "Converting To METIS Graph File..." << endl;
    filename = CudaDataConvert2METIS(filename);

    typedef property<vertex_distance_t, int> VertexProperty;
    typedef property<edge_weight_t, int> EdgeProperty;
    typedef adjacency_list<listS, distributedS<mpi_process_group, vecS>, directedS, VertexProperty, EdgeProperty> Graph;

    cout << "Reading Graph..." << endl;
    ifstream fin(filename.c_str());
    graph::metis_reader reader(fin);

    cout << "Constrcting Graph..." << endl;
    Graph g(reader.begin(), reader.end(), reader.weight_begin(), reader.num_vertices());
    graph_traits<Graph>::vertex_descriptor start = vertex(source, g);

    typedef property_map<Graph, vertex_index_t>::const_type IndexMap;
    typedef iterator_property_map<vector<int>::iterator, IndexMap> DistanceMap;
    vector<int> dist(num_vertices(g), 0);
    DistanceMap distance(dist.begin(), get(vertex_index, g));

    fin.close();

    if (target == -1)
        target = num_vertices(g) - 1;

    cout << "Doing Shortest Alogrithm..." << endl;

    timer time;

    crauser_et_al_shortest_paths(g, start, dummy_property_map(), distance);
    
    PrintResult("Crauser", source, target, dist[target], time.elapsed());

    g.clear();

    string cmd = "del " + filename;
    std::system(cmd.c_str());
}

void test_delta_stepping_dijkstra(string filename, int source, int target)
{    
    cout << "Converting To METIS Graph File..." << endl;
    filename = CudaDataConvert2METIS(filename);

    typedef property<vertex_distance_t, int> VertexProperty;
    typedef property<edge_weight_t, int> EdgeProperty;
    typedef adjacency_list<listS, distributedS<mpi_process_group, vecS>, directedS, VertexProperty, EdgeProperty> Graph;

    cout << "Reading Graph..." << endl;
    ifstream fin(filename.c_str());
    graph::metis_reader reader(fin);

    cout << "Constrcting Graph..." << endl;
    Graph g(reader.begin(), reader.end(), reader.weight_begin(), reader.num_vertices());
    graph_traits<Graph>::vertex_descriptor start = vertex(0, g);

    typedef property_map<Graph, vertex_index_t>::const_type IndexMap;
    typedef iterator_property_map<vector<int>::iterator, IndexMap> DistanceMap;
    vector<int> dist(num_vertices(g), 0);
    DistanceMap distance(dist.begin(), get(vertex_index, g));

    fin.close();

    if (target == -1)
        target = num_vertices(g) - 1;
    
    cout << "Doing Shortest Alogrithm..." << endl;

    timer time;

    delta_stepping_shortest_paths(g, start, dummy_property_map(), distance, get(edge_weight, g));
    
    PrintResult("Delta Stepping", source, target, dist[target], time.elapsed());

    g.clear();

    string cmd = "del " + filename;
    std::system(cmd.c_str());
}

void test_sequential_dijkstra(string filename, int source, int target)
{  
    cout << "Converting To METIS Graph File..." << endl;
    filename = CudaDataConvert2METIS(filename);

    typedef property<vertex_distance_t, int> VertexProperty;
    typedef property<edge_weight_t, int> EdgeProperty;
    typedef adjacency_list<listS, vecS, directedS, VertexProperty , EdgeProperty> Graph;

    cout << "Reading Graph..." << endl;
    ifstream fin(filename.c_str());
    graph::metis_reader reader(fin);

    cout << "Constrcting Graph..." << endl;
    Graph g(reader.begin(), reader.end(), reader.weight_begin(), reader.num_vertices());
    graph_traits<Graph>::vertex_descriptor start = vertex(source, g);

    vector<int> dist(num_vertices(g));

    fin.close();
    
    if (target == -1)
        target = num_vertices(g) - 1;

    cout << "Doing Shortest Alogrithm..." << endl;

    timer time;

    dijkstra_shortest_paths(g, start, distance_map(&dist[0]));
  
    PrintResult("Dijkstra", source, target, dist[target], time.elapsed());

    g.clear();

    string cmd = "del " + filename;
    std::system(cmd.c_str());
}

