#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <omp.h>

using namespace std;

const float INF = numeric_limits<float>::max();
const float EPSILON = 1e-6; 

struct Edge {
    int u, v;
    float weight;
    bool isInsert; 
};

struct Graph {
    int V;
    vector<vector<pair<int, float>>> adj; 
};

struct SSSPTree {
    vector<int> Parent;
    vector<float> Dist;
    vector<bool> Affected_Del;
    vector<bool> Affected;
    vector<vector<int>> Children; 
};

// ---------------------- Algorithm 1: Single Change Update ----------------------
void processSingleChange(Graph& g, SSSPTree& tree, const Edge& change) {
    int u = change.u;
    int v = change.v;
    float w = change.weight;

    if (!change.isInsert) { 
        if (tree.Parent[v] == u) {
            tree.Dist[v] = INF;
            tree.Parent[v] = -1;
            tree.Affected_Del[v] = true;
            tree.Affected[v] = true;
            
            auto it = find(tree.Children[u].begin(), tree.Children[u].end(), v);
            if (it != tree.Children[u].end()) {
                tree.Children[u].erase(it);
            }
            
            auto edge_it = find_if(g.adj[u].begin(), g.adj[u].end(), 
                                 [v](const pair<int, float>& p) { return p.first == v; });
            if (edge_it != g.adj[u].end()) g.adj[u].erase(edge_it);
            
            edge_it = find_if(g.adj[v].begin(), g.adj[v].end(), 
                            [u](const pair<int, float>& p) { return p.first == u; });
            if (edge_it != g.adj[v].end()) g.adj[v].erase(edge_it);
        }
    } else { 
        int x = (tree.Dist[u] < tree.Dist[v]) ? u : v;
        int y = (x == u) ? v : u;
        
        if (tree.Dist[y] > tree.Dist[x] + w + EPSILON) {
            tree.Dist[y] = tree.Dist[x] + w;
            
            if (tree.Parent[y] != -1) {
                auto it = find(tree.Children[tree.Parent[y]].begin(), 
                             tree.Children[tree.Parent[y]].end(), y);
                if (it != tree.Children[tree.Parent[y]].end()) {
                    tree.Children[tree.Parent[y]].erase(it);
                }
            }
            tree.Parent[y] = x;
            tree.Children[x].push_back(y);
            tree.Affected[y] = true;
            
            bool exists = any_of(g.adj[u].begin(), g.adj[u].end(), 
                               [v](const pair<int, float>& p) { return p.first == v; });
            if (!exists) {
                g.adj[u].emplace_back(v, w);
                g.adj[v].emplace_back(u, w);
            }
        }
    }
}

// ---------------------- Algorithm 2: Identify Affected Vertices ----------------------
void identifyAffectedVertices(Graph& g, SSSPTree& tree, const vector<Edge>& changes) {
    for (const auto& change : changes) {
        processSingleChange(g, tree, change);
    }
}

// ---------------------- Algorithm 3: Synchronous Update ----------------------
void updateAffectedVertices(const Graph& g, SSSPTree& tree) {
    bool changed;
    
    do {
        changed = false;
        for (int v = 0; v < g.V; ++v) {
            if (tree.Affected_Del[v]) {
                tree.Affected_Del[v] = false;
                
                for (int c : tree.Children[v]) {
                    tree.Dist[c] = INF;
                    tree.Parent[c] = -1;
                    tree.Affected_Del[c] = true;
                    tree.Affected[c] = true;
                    changed = true;
                }
            }
        }
    } while (changed);
    
    do {
        changed = false;
        for (int v = 0; v < g.V; ++v) {
            if (tree.Affected[v]) {
                tree.Affected[v] = false;
                
                for (auto [n, w] : g.adj[v]) {
                    if (tree.Dist[n] > tree.Dist[v] + w + EPSILON) {
                        tree.Dist[n] = tree.Dist[v] + w;
                        if (tree.Parent[n] != -1) {
                            auto it = find(tree.Children[tree.Parent[n]].begin(), 
                                         tree.Children[tree.Parent[n]].end(), n);
                            if (it != tree.Children[tree.Parent[n]].end()) {
                                tree.Children[tree.Parent[n]].erase(it);
                            }
                        }
                        tree.Parent[n] = v;
                        tree.Children[v].push_back(n);
                        tree.Affected[n] = true;
                        changed = true;
                    }
                    
                    if (tree.Dist[v] > tree.Dist[n] + w + EPSILON) {
                        tree.Dist[v] = tree.Dist[n] + w;
                        if (tree.Parent[v] != -1) {
                            auto it = find(tree.Children[tree.Parent[v]].begin(), 
                                         tree.Children[tree.Parent[v]].end(), v);
                            if (it != tree.Children[tree.Parent[v]].end()) {
                                tree.Children[tree.Parent[v]].erase(it);
                            }
                        }
                        tree.Parent[v] = n;
                        tree.Children[n].push_back(v);
                        tree.Affected[v] = true;
                        changed = true;
                    }
                }
            }
        }
    } while (changed);
}

// ---------------------- Helper Functions ----------------------
Graph readGraph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int V, E;
    file >> V >> E;
    
    Graph g;
    g.V = V;
    g.adj.resize(V);
    
    for (int i = 0; i < E; ++i) {
        int u, v;
        float w;
        file >> u >> v >> w;
        g.adj[u].emplace_back(v, w);
        g.adj[v].emplace_back(u, w);
    }
    
    file.close();
    return g;
}

vector<Edge> readChanges(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int numChanges;
    file >> numChanges;
    
    vector<Edge> changes(numChanges);
    
    for (int i = 0; i < numChanges; ++i) {
        int type, u, v;
        float w;
        file >> type >> u >> v >> w;
        changes[i] = {u, v, w, type == 1};
    }
    
    file.close();
    return changes;
}

SSSPTree initSSSP(const Graph& g, int source) {
    SSSPTree tree;
    tree.Parent.resize(g.V, -1);
    tree.Dist.resize(g.V, INF);
    tree.Affected_Del.resize(g.V, false);
    tree.Affected.resize(g.V, false);
    tree.Children.resize(g.V);
    
    tree.Dist[source] = 0;
    
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    pq.emplace(0, source);
    
    while (!pq.empty()) {
        auto [dist_u, u] = pq.top();
        pq.pop();
        
        if (dist_u > tree.Dist[u]) continue;
        
        for (auto [v, w] : g.adj[u]) {
            if (tree.Dist[v] > tree.Dist[u] + w) {
                tree.Dist[v] = tree.Dist[u] + w;
                tree.Parent[v] = u;
                tree.Children[u].push_back(v);
                pq.emplace(tree.Dist[v], v);
            }
        }
    }
    
    return tree;
}

void printSSSP(const SSSPTree& tree) {
    cout << "Vertex\tDistance\tParent" << endl;
    for (int i = 0; i < tree.Parent.size(); ++i) {
        cout << i << "\t";
        if (tree.Dist[i] == INF) {
            cout << "INF";
        } else {
            cout << tree.Dist[i];
        }
        cout << "\t\t" << tree.Parent[i] << endl;
    }
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <graph_file> <changes_file> <source_vertex>" << endl;
        return 1;
    }
    
    string graphFile = argv[1];
    string changesFile = argv[2];
    int source = stoi(argv[3]);
    
    // Read input
    Graph g = readGraph(graphFile);
    vector<Edge> changes = readChanges(changesFile);
    
    // Initialize SSSP tree
    SSSPTree initialTree = initSSSP(g, source);
    
    cout << "Initial SSSP Tree:" << endl;
    printSSSP(initialTree);
    
    // Update SSSP
    double start = omp_get_wtime();
    
    // Algorithm 2: Identify affected vertices
    identifyAffectedVertices(g, initialTree, changes);
    
    // Algorithm 3: Synchronous update
    updateAffectedVertices(g, initialTree);
    
    double end = omp_get_wtime();
    
    cout << "\nUpdated SSSP Tree:" << endl;
    printSSSP(initialTree);
    ofstream outfile("output.txt");
if (!outfile.is_open()) {
    cerr << "Could not open output.txt for writing\n";
    return 1;
}

outfile << "Vertex\tDistance\tParent\n";
for (int i = 0; i < initialTree.Dist.size(); ++i) {
    outfile << i << "\t";
    if (initialTree.Dist[i] < INF) {
        outfile << initialTree.Dist[i] << "\t";
    } else {
        outfile << "INF\t";
    }
    outfile << initialTree.Parent[i] << "\n";
}

    
    cout << "\nUpdate time: " << (end - start) << " seconds" << endl;
    
    return 0;
}