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

// Performance metrics structure
struct PerformanceMetrics {
    double total_time = 0.0;
    double init_time = 0.0;
    double identify_time = 0.0;
    double update_time = 0.0;
    double io_time = 0.0;
    double speedup = 1.0; // Will be 1.0 for serial execution
};

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
void identifyAffectedVertices(Graph& g, SSSPTree& tree, const vector<Edge>& changes, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    
    for (const auto& change : changes) {
        processSingleChange(g, tree, change);
    }
    
    double end = omp_get_wtime();
    metrics.identify_time = end - start;
}

// ---------------------- Algorithm 3: Synchronous Update ----------------------
void updateAffectedVertices(const Graph& g, SSSPTree& tree, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
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
    
    double end = omp_get_wtime();
    metrics.update_time = end - start;
}

// ---------------------- Helper Functions ----------------------
Graph readGraph(const string& filename, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    
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
    
    double end = omp_get_wtime();
    metrics.io_time += (end - start);
    return g;
}

vector<Edge> readChanges(const string& filename, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    
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
    
    double end = omp_get_wtime();
    metrics.io_time += (end - start);
    return changes;
}

SSSPTree initSSSP(const Graph& g, int source, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    
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
    
    double end = omp_get_wtime();
    metrics.init_time = end - start;
    return tree;
}

void saveResults(const SSSPTree& tree, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    
    ofstream outfile("output.txt");
    if (!outfile.is_open()) {
        cerr << "Could not open output.txt for writing\n";
        return;
    }

    outfile << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < tree.Dist.size(); ++i) {
        outfile << i << "\t";
        if (tree.Dist[i] < INF) {
            outfile << tree.Dist[i] << "\t";
        } else {
            outfile << "INF\t";
        }
        outfile << tree.Parent[i] << "\n";
    }
    
    // Save performance metrics to a separate file
    ofstream metrics_file("performance_metrics.txt");
    if (metrics_file.is_open()) {
        metrics_file << "Performance Metrics:\n";
        metrics_file << "Total Execution Time: " << metrics.total_time << " seconds\n";
        metrics_file << "Initialization Time: " << metrics.init_time << " seconds\n";
        metrics_file << "I/O Time: " << metrics.io_time << " seconds\n";
        metrics_file << "Affected Vertices Identification Time: " << metrics.identify_time << " seconds\n";
        metrics_file << "Update Time: " << metrics.update_time << " seconds\n";
        metrics_file << "Speedup (compared to serial): " << metrics.speedup << "\n";
        metrics_file << "Breakdown:\n";
        metrics_file << "  - Initialization: " << (metrics.init_time / metrics.total_time * 100) << "%\n";
        metrics_file << "  - I/O: " << (metrics.io_time / metrics.total_time * 100) << "%\n";
        metrics_file << "  - Identify Affected: " << (metrics.identify_time / metrics.total_time * 100) << "%\n";
        metrics_file << "  - Update: " << (metrics.update_time / metrics.total_time * 100) << "%\n";
        metrics_file.close();
    }
    
    double end = omp_get_wtime();
    metrics.io_time += (end - start);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <graph_file> <changes_file> <source_vertex>" << endl;
        return 1;
    }
    
    PerformanceMetrics metrics;
    double program_start = omp_get_wtime();
    
    string graphFile = argv[1];
    string changesFile = argv[2];
    int source = stoi(argv[3]);
    
    // Read input
    Graph g = readGraph(graphFile, metrics);
    vector<Edge> changes = readChanges(changesFile, metrics);
    
    // Initialize SSSP tree
    SSSPTree initialTree = initSSSP(g, source, metrics);
    
    // Update SSSP
    // Algorithm 2: Identify affected vertices
    identifyAffectedVertices(g, initialTree, changes, metrics);
    
    // Algorithm 3: Synchronous update
    updateAffectedVertices(g, initialTree, metrics);
    
    // Save results
    saveResults(initialTree, metrics);
    
    double program_end = omp_get_wtime();
    metrics.total_time = program_end - program_start;
    
    // For serial execution, speedup is 1.0 (baseline)
    metrics.speedup = 1.0;
    
    // Print summary to console
    cout << "\nPerformance Summary:\n";
    cout << "Total Execution Time: " << metrics.total_time << " seconds\n";
    cout << "Breakdown:\n";
    cout << "  - Initialization: " << metrics.init_time << "s (" 
         << (metrics.init_time / metrics.total_time * 100) << "%)\n";
    cout << "  - I/O Operations: " << metrics.io_time << "s (" 
         << (metrics.io_time / metrics.total_time * 100) << "%)\n";
    cout << "  - Identify Affected Vertices: " << metrics.identify_time << "s (" 
         << (metrics.identify_time / metrics.total_time * 100) << "%)\n";
    cout << "  - Update Affected Vertices: " << metrics.update_time << "s (" 
         << (metrics.update_time / metrics.total_time * 100) << "%)\n";
    cout <<"  - Speed up:"<<metrics.speedup;
    
    return 0;
}