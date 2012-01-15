#include "CudaDataConvert2METIS.h"
#include "stdafx.h"
#include <fstream>
#include <list>
#include <vector>
#include <string>
using namespace std;

struct Graph
{
	int node, weight;

    Graph(){}
    Graph(int inNode, int w):node(inNode), weight(w){}
};
vector<list<Graph> > g;
int n, m; // n: nodes, m: edges

void ReadGraph(string filename)
{
    ifstream fin(filename.c_str());

    char op;
	string line;

	int inNode, outNode, weight;

	while(fin >> op)
	{
		if(op == 'c')
		{
			getline(fin, line);
		}
		else if (op == 'p')
		{
			string type;
			fin >> type >> n >> m;
			g.resize(n);
		}
		else
		{
			fin >> inNode >> outNode >> weight;

            Graph newNode(inNode - 1, weight);

            g[outNode - 1].push_back(newNode);
		}
	}
}

void ReadRandomGraph(string filename)
{
    ifstream fin(filename.c_str());

    fin >> n;
    g.resize(n);

    m = 0;

    int outNode, inNode, weight;
    while (fin >> outNode)
    {
        if (outNode == -1)
            return;

        m++;

        fin >> inNode >> weight;

        Graph newNode(inNode, weight);

        g[outNode].push_back(newNode);
    }
}

void OutputMETISGraph(string filename)
{
    ofstream fout(filename.c_str());

    fout << n << ' ' << m << ' ' << 1 << endl;
	
    for(int i = 0; i < n; i++)
	{
        for(list<Graph>::iterator iter = g[i].begin(); iter != g[i].end(); iter++)
		{
            fout << iter->node + 1 << ' ' << iter->weight << ' ';
		}
		fout << endl;
	}

    g.clear();
}

string CudaDataConvert2METIS(string filename)
{
    ReadGraph(filename);
    string NewFilename = filename + ".metis";
    OutputMETISGraph(NewFilename);
	
    return NewFilename;
}