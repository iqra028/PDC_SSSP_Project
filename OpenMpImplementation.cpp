#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <omp.h>
#include <algorithm>
#include <iomanip>

using namespace std;

const float INF = numeric_limits<float>::max();
const float EPSILON = 1e-6; 

struct PerformanceMetrics {
    double total_time = 0.0;
    double init_time = 0.0;
    double identify_time = 0.0;
    double update_time = 0.0;
    double sync_time = 0.0;
    double async_time = 0.0;
    double io_time = 0.0;
    double speedup = 1.0;
    int num_threads = 1;
    string update_method = "serial";
    
    void print() const {
        cout << "\nPerformance Metrics:\n";
        cout << "----------------------------------------\n";
        cout << "Total Execution Time: " << total_time << " seconds\n";
        cout << "Initialization Time: " << init_time << " seconds\n";
        cout << "I/O Time: " << io_time << " seconds\n";
        cout << "Affected Vertices Identification Time: " << identify_time << " seconds\n";
        cout << "Update Time: " << update_time << " seconds\n";
        if (update_method == "both") {
            cout << "  - Synchronous: " << sync_time << " seconds\n";
            cout << "  - Asynchronous: " << async_time << " seconds\n";
        }
        cout << "Speedup: " << speedup << "x\n";
        cout << "Number of Threads: " << num_threads << "\n";
        cout << "Update Method: " << update_method << "\n";
        cout << "Time Breakdown:\n";
        cout << "  - Initialization: " << (init_time/total_time)*100 << "%\n";
        cout << "  - I/O: " << (io_time/total_time)*100 << "%\n";
        cout << "  - Identify Affected: " << (identify_time/total_time)*100 << "%\n";
        cout << "  - Update: " << (update_time/total_time)*100 << "%\n";
        cout << "----------------------------------------\n";
    }
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
                #pragma omp critical
                tree.Children[u].erase(it);
            }
            
            auto edge_it = find_if(g.adj[u].begin(), g.adj[u].end(), 
                                 [v](const pair<int, float>& p) { return p.first == v; });
            if (edge_it != g.adj[u].end()) {
                #pragma omp critical
                g.adj[u].erase(edge_it);
            }
            
            edge_it = find_if(g.adj[v].begin(), g.adj[v].end(), 
                            [u](const pair<int, float>& p) { return p.first == u; });
            if (edge_it != g.adj[v].end()) {
                #pragma omp critical
                g.adj[v].erase(edge_it);
            }
        }
    } else {
        int x = (tree.Dist[u] < tree.Dist[v]) ? u : v;
        int y = (x == u) ? v : u;
        
        if (tree.Dist[y] > tree.Dist[x] + w + EPSILON) {
            #pragma omp atomic write
            tree.Dist[y] = tree.Dist[x] + w;
            
            if (tree.Parent[y] != -1) {
                auto it = find(tree.Children[tree.Parent[y]].begin(), 
                             tree.Children[tree.Parent[y]].end(), y);
                if (it != tree.Children[tree.Parent[y]].end()) {
                    #pragma omp critical
                    tree.Children[tree.Parent[y]].erase(it);
                }
            }
            
            tree.Parent[y] = x; 
            #pragma omp critical
            tree.Children[x].push_back(y);
            tree.Affected[y] = true;
            
            bool exists = any_of(g.adj[u].begin(), g.adj[u].end(), 
                               [v](const pair<int, float>& p) { return p.first == v; });
            if (!exists) {
                #pragma omp critical
                {
                    g.adj[u].emplace_back(v, w);
                    g.adj[v].emplace_back(u, w);
                }
            }
        }
    }
}
// ---------------------- Algorithm 2: Identify Affected Vertices ----------------------
void identifyAffectedVertices(Graph& g, SSSPTree& tree, const vector<Edge>& changes, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); ++i) {
        processSingleChange(g, tree, changes[i]);
    }
    
    double end = omp_get_wtime();
    metrics.identify_time = end - start;
}

// ---------------------- Algorithm 3: Synchronous Update ----------------------
void updateAffectedVertices(const Graph& g, SSSPTree& tree, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    bool changed;
    
    // First phase: handle deletions
    vector<bool> local_changed(omp_get_max_threads(), false);
    do {
        changed = false;
        #pragma omp parallel reduction(||:changed)
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic, 64)
            for (int v = 0; v < g.V; ++v) {
                if (tree.Affected_Del[v]) {
                    tree.Affected_Del[v] = false;
                    
                    for (int c : tree.Children[v]) {
                        #pragma omp atomic write
                        tree.Dist[c] = INF;
                        tree.Parent[c] = -1;
                        tree.Affected_Del[c] = true;
                        tree.Affected[c] = true;
                        local_changed[tid] = true;
                    }
                }
            }
            changed = changed || local_changed[tid];
            local_changed[tid] = false;
        }
    } while (changed);
    
    // Second phase: handle updates
    vector<vector<int>> local_updates(omp_get_max_threads());
    do {
        changed = false;
        #pragma omp parallel reduction(||:changed)
        {
            int tid = omp_get_thread_num();
            auto& my_updates = local_updates[tid];
            
            #pragma omp for schedule(dynamic, 64)
            for (int v = 0; v < g.V; ++v) {
                if (tree.Affected[v]) {
                    my_updates.push_back(v);
                }
            }
            
            for (int v : my_updates) {
                tree.Affected[v] = false;
                
                for (auto [n, w] : g.adj[v]) {
                    bool updated = false;
                    
                    if (tree.Dist[n] > tree.Dist[v] + w + EPSILON) {
                        #pragma omp atomic write
                        tree.Dist[n] = tree.Dist[v] + w;
                        
                        int old_parent = tree.Parent[n];
                        if (old_parent != -1) {
                            #pragma omp critical
                            {
                                auto it = find(tree.Children[old_parent].begin(), 
                                             tree.Children[old_parent].end(), n);
                                if (it != tree.Children[old_parent].end()) {
                                    tree.Children[old_parent].erase(it);
                                }
                            }
                        }
                        
                        tree.Parent[n] = v; 
                        #pragma omp critical
                        tree.Children[v].push_back(n);
                        tree.Affected[n] = true;
                        updated = local_changed[tid] = true;
                    }
                    
                    if (tree.Dist[v] > tree.Dist[n] + w + EPSILON) {
                        #pragma omp atomic write
                        tree.Dist[v] = tree.Dist[n] + w;
                        
                        int old_parent = tree.Parent[v];
                        if (old_parent != -1) {
                            #pragma omp critical
                            {
                                auto it = find(tree.Children[old_parent].begin(), 
                                             tree.Children[old_parent].end(), v);
                                if (it != tree.Children[old_parent].end()) {
                                    tree.Children[old_parent].erase(it);
                                }
                            }
                        }
                        
                        tree.Parent[v] = n;  
                        #pragma omp critical
                        tree.Children[n].push_back(v);
                        tree.Affected[v] = true;
                        updated = local_changed[tid] = true;
                    }
                }
            }
            my_updates.clear();
            changed = changed || local_changed[tid];
            local_changed[tid] = false;
        }
    } while (changed);
    
    double end = omp_get_wtime();
    metrics.sync_time = end - start;
    metrics.update_time = metrics.sync_time;
}
// ---------------------- Algorithm 4: Asynchronous Update ----------------------

void asyncUpdateAffectedVertices(const Graph& g, SSSPTree& tree, int async_level, PerformanceMetrics& metrics) {
    double start = omp_get_wtime();
    bool changed;
    vector<vector<int>> local_queues(omp_get_max_threads());
    vector<bool> local_changed(omp_get_max_threads(), false);
    
    // First phase: handle deletions
    do {
        changed = false;
        #pragma omp parallel reduction(||:changed)
        {
            int tid = omp_get_thread_num();
            auto& q = local_queues[tid];
            
            #pragma omp for schedule(dynamic, 64) nowait
            for (int v = 0; v < g.V; ++v) {
                if (tree.Affected_Del[v]) {
                    q.push_back(v);
                }
            }
            
            while (!q.empty()) {
                int v = q.back();
                q.pop_back();
                
                // Use simple assignment instead of atomic write for boolean
                tree.Affected_Del[v] = false;
                
                for (int c : tree.Children[v]) {
                    // Use proper atomic updates
                    #pragma omp atomic write
                    tree.Dist[c] = INF;
                    tree.Parent[c] = -1;  // No need for atomic for single assignment
                    tree.Affected_Del[c] = true;
                    tree.Affected[c] = true;
                    local_changed[tid] = true;
                    
                    if (async_level > 1) {
                        q.push_back(c);
                    }
                }
            }
            changed = changed || local_changed[tid];
            local_changed[tid] = false;
        }
    } while (changed);
    
    // Second phase: handle updates
    do {
        changed = false;
        #pragma omp parallel reduction(||:changed)
        {
            int tid = omp_get_thread_num();
            auto& q = local_queues[tid];
            int level = 0;
            
            #pragma omp for schedule(dynamic, 64) nowait
            for (int v = 0; v < g.V; ++v) {
                if (tree.Affected[v]) {
                    q.push_back(v);
                }
            }
            
            while (!q.empty() && level < async_level) {
                int v = q.back();
                q.pop_back();
                
                tree.Affected[v] = false;
                
                for (auto [n, w] : g.adj[v]) {
                    bool updated = false;
                    
                    if (tree.Dist[n] > tree.Dist[v] + w + EPSILON) {
                        #pragma omp atomic write
                        tree.Dist[n] = tree.Dist[v] + w;
                        
                        int old_parent = tree.Parent[n];
                        if (old_parent != -1) {
                            #pragma omp critical
                            {
                                auto it = find(tree.Children[old_parent].begin(), 
                                            tree.Children[old_parent].end(), n);
                                if (it != tree.Children[old_parent].end()) {
                                    tree.Children[old_parent].erase(it);
                                }
                            }
                        }
                        
                        tree.Parent[n] = v;  // No need for atomic for single assignment
                        #pragma omp critical
                        tree.Children[v].push_back(n);
                        tree.Affected[n] = true;
                        updated = local_changed[tid] = true;
                    }
                    
                    if (tree.Dist[v] > tree.Dist[n] + w + EPSILON) {
                        #pragma omp atomic write
                        tree.Dist[v] = tree.Dist[n] + w;
                        
                        int old_parent = tree.Parent[v];
                        if (old_parent != -1) {
                            #pragma omp critical
                            {
                                auto it = find(tree.Children[old_parent].begin(), 
                                            tree.Children[old_parent].end(), v);
                                if (it != tree.Children[old_parent].end()) {
                                    tree.Children[old_parent].erase(it);
                                }
                            }
                        }
                        
                        tree.Parent[v] = n;  // No need for atomic for single assignment
                        #pragma omp critical
                        tree.Children[n].push_back(v);
                        tree.Affected[v] = true;
                        updated = local_changed[tid] = true;
                    }
                    
                    if (updated && level + 1 < async_level) {
                        q.push_back(n);
                    }
                }
                level++;
            }
            changed = changed || local_changed[tid];
            local_changed[tid] = false;
        }
    } while (changed);
    
    double end = omp_get_wtime();
    metrics.async_time = end - start;
    metrics.update_time = metrics.async_time;
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

void saveResults(const SSSPTree& tree, const PerformanceMetrics& metrics, const string& suffix = "") {
    string filename = "output" + suffix + ".txt";
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Could not open " << filename << " for writing\n";
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
    outfile.close();
    
    filename = "metrics" + suffix + ".txt";
    ofstream metrics_file(filename);
    if (metrics_file.is_open()) {
        metrics_file << "Performance Metrics (" << metrics.update_method << "):\n";
        metrics_file << "Total Execution Time: " << metrics.total_time << " seconds\n";
        metrics_file << "Initialization Time: " << metrics.init_time << " seconds\n";
        metrics_file << "I/O Time: " << metrics.io_time << " seconds\n";
        metrics_file << "Affected Vertices Identification Time: " << metrics.identify_time << " seconds\n";
        metrics_file << "Update Time: " << metrics.update_time << " seconds\n";
        if (metrics.update_method == "both") {
            metrics_file << "  - Synchronous: " << metrics.sync_time << " seconds\n";
            metrics_file << "  - Asynchronous: " << metrics.async_time << " seconds\n";
        }
        metrics_file << "Speedup: " << metrics.speedup << "x\n";
        metrics_file << "Number of Threads: " << metrics.num_threads << "\n";
        metrics_file << "Update Method: " << metrics.update_method << "\n";
        metrics_file.close();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <graph_file> <changes_file> <source_vertex>" << endl;
        return 1;
    }
    
    PerformanceMetrics metrics;
    double program_start = omp_get_wtime();
    metrics.num_threads = omp_get_max_threads();
    
    string graphFile = argv[1];
    string changesFile = argv[2];
    int source = stoi(argv[3]);
    int async_level = 3; 
    
    // Read input
    Graph g = readGraph(graphFile, metrics);
    vector<Edge> changes = readChanges(changesFile, metrics);
    
    // Initialize SSSP tree
    SSSPTree initialTree = initSSSP(g, source, metrics);
    
    SSSPTree syncTree = initialTree;
    SSSPTree asyncTree = initialTree;
    
    // 1. Synchronous update
    // Algorithm 2
    identifyAffectedVertices(g, syncTree, changes, metrics);
    //Algorithm 3
    updateAffectedVertices(g, syncTree, metrics);
    double sync_update_time = metrics.update_time;
    
    // 2. Asynchronous update
    metrics.update_time = 0; 
    // Algorithm 3
    identifyAffectedVertices(g, asyncTree, changes, metrics);
    // Algorithm 4
    asyncUpdateAffectedVertices(g, asyncTree, async_level, metrics);
    double async_update_time = metrics.update_time;
    
    double program_end = omp_get_wtime();
    metrics.total_time = program_end - program_start;
    double serial_time = 0.290735; 
    metrics.speedup = serial_time / metrics.total_time;
    metrics.update_method = "both"; 
    
    saveResults(syncTree, metrics, "_sync");
    saveResults(asyncTree, metrics, "_async");
    
    metrics.print();
    
    return 0;
}