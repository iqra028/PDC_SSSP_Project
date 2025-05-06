#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <metis.h>
#include <cmath>

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
    double partition_time = 0.0;
    double comm_time = 0.0;
    double speedup = 1.0;
    int num_threads = 1;
    int mpi_rank = 0;
    string update_method = "serial";

    void print() const {
        cout << "\nPerformance Metrics (Rank " << mpi_rank << "):\n";
        cout << "----------------------------------------\n";
        cout << "Total Execution Time: " << total_time << " seconds\n";
        cout << "Initialization Time: " << init_time << " seconds\n";
        cout << "I/O Time: " << io_time << " seconds\n";
        cout << "Partitioning Time: " << partition_time << " seconds\n";
        cout << "Communication Time: " << comm_time << " seconds\n";
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
        cout << "  - Partitioning: " << (partition_time/total_time)*100 << "%\n";
        cout << "  - Communication: " << (comm_time/total_time)*100 << "%\n";
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

bool anyTrue(const vector<bool>& vec) {
    for (bool b : vec) {
        if (b) return true;
    }
    return false;
}

void partitionGraphWithMetis(const Graph& g, int nparts, vector<int>& part, PerformanceMetrics& metrics) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Starting METIS partitioning for " << g.V << " vertices and " << nparts << " parts" << endl << flush;

    double start = omp_get_wtime();

    idx_t nvtxs = g.V;
    idx_t ncon = 1;
    vector<idx_t> xadj(nvtxs + 1, 0);
    vector<idx_t> adjncy;

    for (int u = 0; u < g.V; ++u) {
        xadj[u] = adjncy.size();
        for (auto [v, w] : g.adj[u]) {
            adjncy.push_back(v);
        }
    }
    xadj[nvtxs] = adjncy.size();

    idx_t nparts_metis = nparts;
    idx_t objval;
    part.resize(nvtxs);

    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,
                                  &nparts_metis, nullptr, nullptr, nullptr, &objval, part.data());

    if (ret != METIS_OK) {
        cerr << "[Rank " << rank << "] METIS partitioning failed" << endl << flush;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double end = omp_get_wtime();
    metrics.partition_time = end - start;
    cout << "[Rank " << rank << "] METIS partitioning completed in " << metrics.partition_time << " seconds" << endl << flush;
}

Graph buildLocalGraph(const Graph& global_g, const vector<int>& part, int rank, vector<int>& local_to_global) {
    cout << "[Rank " << rank << "] Building local graph" << endl << flush;

    Graph local_g;
    vector<int> global_to_local(global_g.V, -1);

    int local_V = 0;
    for (int v = 0; v < global_g.V; ++v) {
        if (part[v] == rank) {
            global_to_local[v] = local_V++;
            local_to_global.push_back(v);
        }
    }

    local_g.V = local_V;
    local_g.adj.resize(local_V);

    for (int v = 0; v < global_g.V; ++v) {
        if (part[v] == rank) {
            int local_v = global_to_local[v];
            for (auto [u, w] : global_g.adj[v]) {
                int local_u = global_to_local[u];
                if (local_u != -1) {
                    local_g.adj[local_v].emplace_back(local_u, w);
                }
            }
        }
    }

    cout << "[Rank " << rank << "] Local graph built with " << local_V << " vertices" << endl << flush;
    for (int u = 0; u < local_g.V; ++u) {
        cout << "[Rank " << rank << "] Local vertex " << u << " (global " << local_to_global[u]
             << ") has " << local_g.adj[u].size() << " edges" << endl << flush;
    }
    return local_g;
}

void identifyBoundaryVertices(const Graph& global_g, const vector<int>& part, int rank,
                             vector<int>& boundary_vertices, vector<vector<int>>& vertex_to_procs) {
    cout << "[Rank " << rank << "] Identifying boundary vertices" << endl << flush;

    vertex_to_procs.resize(global_g.V);
    for (int v = 0; v < global_g.V; ++v) {
        if (part[v] == rank) {
            for (auto [u, w] : global_g.adj[v]) {
                if (part[u] != rank) {
                    boundary_vertices.push_back(v);
                    vertex_to_procs[v].push_back(part[u]);
                    vertex_to_procs[u].push_back(rank);
                }
            }
        }
    }

    for (auto& procs : vertex_to_procs) {
        sort(procs.begin(), procs.end());
        procs.erase(unique(procs.begin(), procs.end()), procs.end());
    }

    cout << "[Rank " << rank << "] Found " << boundary_vertices.size() << " boundary vertices" << endl << flush;
    for (int v : boundary_vertices) {
        cout << "[Rank " << rank << "] Boundary vertex " << v << " connects to partitions: ";
        for (int p : vertex_to_procs[v]) {
            cout << p << " ";
        }
        cout << endl << flush;
    }
}

void processSingleChange(Graph& g, SSSPTree& tree, const Edge& change, const vector<int>& global_to_local) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int u = global_to_local[change.u] != -1 ? global_to_local[change.u] : -1;
    int v = global_to_local[change.v] != -1 ? global_to_local[change.v] : -1;

    if (u == -1 && v == -1) {
        cout << "[Rank " << rank << "] Skipping change (" << change.u << ", " << change.v << ") - neither vertex in partition" << endl << flush;
        return;
    }

    cout << "[Rank " << rank << "] Processing change (" << change.u << ", " << change.v << ", " << change.weight
         << ", " << (change.isInsert ? "insert" : "delete") << ")" << endl << flush;

    float w = change.weight;

    if (!change.isInsert) {
        if (u != -1 && v != -1 && tree.Parent[v] == u) {
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
            cout << "[Rank " << rank << "] Marked vertex " << change.v << " as affected due to edge deletion" << endl << flush;
        }
    } else {
        if (u != -1 && v != -1) {
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
                cout << "[Rank " << rank << "] Updated vertex " << change.v << " distance to " << tree.Dist[y]
                     << " and marked as affected" << endl << flush;
            }
        }
        if (u == -1) {
            int local_v = v;
            tree.Affected[local_v] = true;
            cout << "[Rank " << rank << "] Marked vertex " << change.v << " as affected (boundary)" << endl << flush;
        } else if (v == -1) {
            int local_u = u;
            tree.Affected[local_u] = true;
            cout << "[Rank " << rank << "] Marked vertex " << change.u << " as affected (boundary)" << endl << flush;
        }
    }
}

void identifyAffectedVertices(Graph& g, SSSPTree& tree, const vector<Edge>& changes,
                             PerformanceMetrics& metrics, const vector<int>& global_to_local) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Identifying affected vertices for " << changes.size() << " changes" << endl << flush;

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); ++i) {
        processSingleChange(g, tree, changes[i], global_to_local);
    }

    for (int v = 0; v < tree.Affected.size(); ++v) {
        if (tree.Affected[v] || tree.Affected_Del[v]) {
            cout << "[Rank " << rank << "] Vertex " << v << " is affected: Affected=" << tree.Affected[v]
                 << ", Affected_Del=" << tree.Affected_Del[v] << endl << flush;
        }
    }

    double end = omp_get_wtime();
    metrics.identify_time = end - start;
    cout << "[Rank " << rank << "] Affected vertices identification took " << metrics.identify_time << " seconds" << endl << flush;
}

void updateAffectedVertices(const Graph& g, SSSPTree& tree, PerformanceMetrics& metrics) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Starting synchronous update" << endl << flush;

    double start = omp_get_wtime();
    bool changed;
    int iteration = 0;
    vector<bool> local_changed(omp_get_max_threads(), false);

    do {
        changed = false;
        iteration++;
        cout << "[Rank " << rank << "] Synchronous update iteration " << iteration << endl << flush;

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

    vector<vector<int>> local_updates(omp_get_max_threads());
    do {
        changed = false;
        iteration++;
        cout << "[Rank " << rank << "] Synchronous update iteration " << iteration << " (distance updates)" << endl << flush;

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
                        cout << "[Rank " << rank << "] Updated vertex " << n << " distance to " << tree.Dist[n] << endl << flush;
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
                        cout << "[Rank " << rank << "] Updated vertex " << v << " distance to " << tree.Dist[v] << endl << flush;
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
    cout << "[Rank " << rank << "] Synchronous update completed in " << metrics.sync_time << " seconds" << endl << flush;
}

void asyncUpdateAffectedVertices(const Graph& g, SSSPTree& tree, int async_level, PerformanceMetrics& metrics) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Starting asynchronous update with level " << async_level << endl << flush;

    double start = omp_get_wtime();
    bool changed;
    int iteration = 0;
    vector<vector<int>> local_queues(omp_get_max_threads());
    vector<bool> local_changed(omp_get_max_threads(), false);

    do {
        changed = false;
        iteration++;
        cout << "[Rank " << rank << "] Asynchronous update iteration " << iteration << " (deletion phase)" << endl << flush;

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
                tree.Affected_Del[v] = false;
                for (int c : tree.Children[v]) {
                    #pragma omp atomic write
                    tree.Dist[c] = INF;
                    tree.Parent[c] = -1;
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

    do {
        changed = false;
        iteration++;
        cout << "[Rank " << rank << "] Asynchronous update iteration " << iteration << " (distance updates)" << endl << flush;

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
                        tree.Parent[n] = v;
                        #pragma omp critical
                        tree.Children[v].push_back(n);
                        tree.Affected[n] = true;
                        updated = local_changed[tid] = true;
                        cout << "[Rank " << rank << "] Async updated vertex " << n << " distance to " << tree.Dist[n] << endl << flush;
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
                        cout << "[Rank " << rank << "] Async updated vertex " << v << " distance to " << tree.Dist[v] << endl << flush;
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
    cout << "[Rank " << rank << "] Asynchronous update completed in " << metrics.async_time << " seconds" << endl << flush;
}

Graph readGraph(const string& filename, PerformanceMetrics& metrics) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Reading graph from " << filename << endl << flush;

    double start = omp_get_wtime();

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "[Rank " << rank << "] Error opening file: " << filename << endl << flush;
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    cout << "[Rank " << rank << "] Graph read: " << V << " vertices, " << E << " edges, took " << (end - start) << " seconds" << endl << flush;

    for (int u = 0; u < g.V; ++u) {
        for (auto [v, w] : g.adj[u]) {
            cout << "[Rank " << rank << "] Edge " << u << " -> " << v << " (weight " << w << ")" << endl << flush;
        }
    }
    return g;
}

vector<Edge> readChanges(const string& filename, PerformanceMetrics& metrics) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Reading changes from " << filename << endl << flush;

    double start = omp_get_wtime();

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "[Rank " << rank << "] Error opening file: " << filename << endl << flush;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numChanges;
    file >> numChanges;

    vector<Edge> changes(numChanges);
    for (int i = 0; i < numChanges; ++i) {
        int type, u, v;
        float w;
        file >> type >> u >> v >> w;
        changes[i] = {u, v, w, type == 1};
        cout << "[Rank " << rank << "] Change " << i << ": type=" << type << ", u=" << u << ", v=" << v << ", w=" << w << endl << flush;
    }

    file.close();

    double end = omp_get_wtime();
    metrics.io_time += (end - start);
    cout << "[Rank " << rank << "] Read " << numChanges << " changes, took " << (end - start) << " seconds" << endl << flush;
    return changes;
}

SSSPTree initSSSP(const Graph& g, int source, PerformanceMetrics& metrics) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "[Rank " << rank << "] Initializing SSSP tree with source " << source << endl << flush;

    double start = omp_get_wtime();

    SSSPTree tree;
    tree.Parent.resize(g.V, -1);
    tree.Dist.resize(g.V, INF);
    tree.Affected_Del.resize(g.V, false);
    tree.Affected.resize(g.V, false);
    tree.Children.resize(g.V);

    if (source >= 0 && source < g.V) {
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
    }

    for (int v = 0; v < g.V; ++v) {
        if (tree.Dist[v] < INF) {
            cout << "[Rank " << rank << "] Vertex " << v << ": dist=" << tree.Dist[v]
                 << ", parent=" << tree.Parent[v] << endl << flush;
        }
    }

    double end = omp_get_wtime();
    metrics.init_time = end - start;
    cout << "[Rank " << rank << "] SSSP tree initialized in " << metrics.init_time << " seconds" << endl << flush;
    return tree;
}

void communicateBoundaryUpdates(SSSPTree& tree, const vector<int>& boundary_vertices,
                               const vector<vector<int>>& vertex_to_procs,
                               const vector<int>& local_to_global,
                               const vector<int>& global_to_local,
                               PerformanceMetrics& metrics) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cout << "[Rank " << rank << "] Starting boundary vertex communication" << endl << flush;

    double start = omp_get_wtime();

    vector<MPI_Request> requests;
    vector<vector<float>> send_buffers(size);
    vector<vector<float>> recv_buffers(size);
    vector<int> send_sizes(size, 0);
    vector<int> recv_sizes(size, 0);

    for (int global_v : boundary_vertices) {
        int local_v = global_to_local[global_v];
        if (local_v != -1) {
            for (int dest_rank : vertex_to_procs[global_v]) {
                send_buffers[dest_rank].push_back(static_cast<float>(global_v));
                send_buffers[dest_rank].push_back(tree.Dist[local_v]);
                send_buffers[dest_rank].push_back(static_cast<float>(tree.Parent[local_v] == -1 ? -1 : local_to_global[tree.Parent[local_v]]));
                send_sizes[dest_rank] += 3;
            }
        }
    }

    vector<MPI_Request> size_requests;
    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;
        MPI_Request req;
        MPI_Isend(&send_sizes[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD, &req);
        size_requests.push_back(req);
        MPI_Irecv(&recv_sizes[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD, &req);
        size_requests.push_back(req);
    }

    MPI_Waitall(size_requests.size(), size_requests.data(), MPI_STATUSES_IGNORE);

    for (int i = 0; i < size; ++i) {
        if (i != rank && send_sizes[i] > 0) {
            cout << "[Rank " << rank << "] Sending " << send_sizes[i] << " floats to rank " << i << endl << flush;
        }
        if (i != rank && recv_sizes[i] > 0) {
            cout << "[Rank " << rank << "] Receiving " << recv_sizes[i] << " floats from rank " << i << endl << flush;
        }
    }

    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;
        recv_buffers[i].resize(recv_sizes[i]);
    }

    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;
        if (!send_buffers[i].empty()) {
            MPI_Request req;
            MPI_Isend(send_buffers[i].data(), send_buffers[i].size(), MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        if (recv_sizes[i] > 0) {
            MPI_Request req;
            MPI_Irecv(recv_buffers[i].data(), recv_sizes[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }

    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    bool changed = false;
    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;
        for (size_t j = 0; j < recv_buffers[i].size(); j += 3) {
            int global_v = static_cast<int>(recv_buffers[i][j]);
            float dist = recv_buffers[i][j + 1];
            int parent = static_cast<int>(recv_buffers[i][j + 2]);
            int local_v = global_to_local[global_v];
            if (local_v != -1 && dist < tree.Dist[local_v] - EPSILON) {
                tree.Dist[local_v] = dist;
                tree.Parent[local_v] = parent == -1 ? -1 : global_to_local[parent];
                tree.Affected[local_v] = true;
                changed = true;
                cout << "[Rank " << rank << "] Updated vertex " << global_v << " from rank " << i
                     << ": dist=" << dist << ", parent=" << parent << endl << flush;
            }
        }
    }

    int local_changed = changed ? 1 : 0;
    int global_changed;
    MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (global_changed) {
        cout << "[Rank " << rank << "] Global changes detected, marking boundary vertices as affected" << endl << flush;
        for (int global_v : boundary_vertices) {
            int local_v = global_to_local[global_v];
            if (local_v != -1) {
                tree.Affected[local_v] = true;
            }
        }
    }

    double end = omp_get_wtime();
    metrics.comm_time += (end - start);
    cout << "[Rank " << rank << "] Boundary communication completed in " << (end - start) << " seconds" << endl << flush;
}

void saveResults(const SSSPTree& tree, const PerformanceMetrics& metrics,
                const vector<int>& local_to_global, const string& suffix = "") {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string filename = "output_rank" + to_string(rank) + suffix + ".txt";
    cout << "[Rank " << rank << "] Saving results to " << filename << endl << flush;

    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "[Rank " << rank << "] Could not open " << filename << " for writing" << endl << flush;
        return;
    }

    outfile << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < tree.Dist.size(); ++i) {
        int global_v = local_to_global[i];
        outfile << global_v << "\t";
        if (tree.Dist[i] < INF) {
            outfile << tree.Dist[i] << "\t";
        } else {
            outfile << "INF\t";
        }
        outfile << (tree.Parent[i] == -1 ? -1 : local_to_global[tree.Parent[i]]) << "\n";
    }
    outfile.close();

    filename = "metrics_rank" + to_string(rank) + suffix + ".txt";
    cout << "[Rank " << rank << "] Saving metrics to " << filename << endl << flush;

    ofstream metrics_file(filename);
    if (metrics_file.is_open()) {
        metrics_file << "Performance Metrics (Rank " << rank << ", " << metrics.update_method << "):\n";
        metrics_file << "Total Execution Time: " << metrics.total_time << " seconds\n";
        metrics_file << "Initialization Time: " << metrics.init_time << " seconds\n";
        metrics_file << "I/O Time: " << metrics.io_time << " seconds\n";
        metrics_file << "Partitioning Time: " << metrics.partition_time << " seconds\n";
        metrics_file << "Communication Time: " << metrics.comm_time << " seconds\n";
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

void combineOutputs(const SSSPTree& syncTree, const SSSPTree& asyncTree,
                   const vector<int>& local_to_global, int global_V) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cout << "[Rank " << rank << "] Combining outputs" << endl << flush;

    vector<float> local_sync_data, local_async_data;
    for (size_t i = 0; i < local_to_global.size(); ++i) {
        int global_v = local_to_global[i];
        local_sync_data.push_back(static_cast<float>(global_v));
        local_sync_data.push_back(syncTree.Dist[i]);
        local_sync_data.push_back(syncTree.Parent[i] == -1 ? -1 : local_to_global[syncTree.Parent[i]]);
        local_async_data.push_back(static_cast<float>(global_v));
        local_async_data.push_back(asyncTree.Dist[i]);
        local_async_data.push_back(asyncTree.Parent[i] == -1 ? -1 : local_to_global[asyncTree.Parent[i]]);
    }

    int local_size = local_sync_data.size();
    vector<int> recv_counts(size);
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    int total_size = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total_size;
            total_size += recv_counts[i];
        }
    }

    vector<float> global_sync_data(total_size), global_async_data(total_size);
    MPI_Gatherv(local_sync_data.data(), local_size, MPI_FLOAT,
                global_sync_data.data(), recv_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_async_data.data(), local_size, MPI_FLOAT,
                global_async_data.data(), recv_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        string sync_filename = "combined_output_sync.txt";
        ofstream sync_out(sync_filename);
        if (!sync_out.is_open()) {
            cerr << "[Rank " << rank << "] Error: Could not open " << sync_filename << " for writing" << endl << flush;
        } else {
            sync_out << "Vertex\tDistance\tParent\n";
            vector<tuple<int, float, int>> sync_entries;
            for (size_t i = 0; i < global_sync_data.size(); i += 3) {
                int vertex = static_cast<int>(global_sync_data[i]);
                float dist = global_sync_data[i + 1];
                int parent = static_cast<int>(global_sync_data[i + 2]);
                sync_entries.emplace_back(vertex, dist, parent);
            }
            sort(sync_entries.begin(), sync_entries.end());
            for (const auto& [vertex, dist, parent] : sync_entries) {
                sync_out << vertex << "\t";
                if (dist < INF) {
                    sync_out << dist << "\t";
                } else {
                    sync_out << "INF\t";
                }
                sync_out << parent << "\n";
            }
            sync_out.close();
            cout << "[Rank " << rank << "] Combined synchronous output written to " << sync_filename << endl << flush;
        }

        string async_filename = "combined_output_async.txt";
        ofstream async_out(async_filename);
        if (!async_out.is_open()) {
            cerr << "[Rank " << rank << "] Error: Could not open " << async_filename << " for writing" << endl << flush;
        } else {
            async_out << "Vertex\tDistance\tParent\n";
            vector<tuple<int, float, int>> async_entries;
            for (size_t i = 0; i < global_async_data.size(); i += 3) {
                int vertex = static_cast<int>(global_async_data[i]);
                float dist = global_async_data[i + 1];
                int parent = static_cast<int>(global_async_data[i + 2]);
                async_entries.emplace_back(vertex, dist, parent);
            }
            sort(async_entries.begin(), async_entries.end());
            for (const auto& [vertex, dist, parent] : async_entries) {
                async_out << vertex << "\t";
                if (dist < INF) {
                    async_out << dist << "\t";
                } else {
                    async_out << "INF\t";
                }
                async_out << parent << "\n";
            }
            async_out.close();
            cout << "[Rank " << rank << "] Combined asynchronous output written to " << async_filename << endl << flush;
        }
    }
}

bool compareWithReferenceOutput(const SSSPTree& syncTree, const SSSPTree& asyncTree,
                               const vector<int>& local_to_global, int global_V,
                               const string& referenceFile) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cout << "[Rank " << rank << "] Starting comparison with reference file: " << referenceFile << endl << flush;

    bool local_valid = true;
    vector<string> local_sync_errors, local_async_errors;

    {
        ifstream refFile(referenceFile);
        if (!refFile.is_open()) {
            cerr << "[Rank " << rank << "] Error: Could not open reference file " << referenceFile << endl << flush;
            local_valid = false;
        } else {
            string line;
            getline(refFile, line);

            vector<float> refDistances(global_V, INF);
            vector<int> refParents(global_V, -1);
            while (getline(refFile, line)) {
                istringstream iss(line);
                int vertex;
                string distStr;
                int parent;
                if (!(iss >> vertex >> distStr >> parent)) {
                    cerr << "[Rank " << rank << "] Error: Invalid format in reference file at line: " << line << endl << flush;
                    local_valid = false;
                    break;
                }
                if (vertex >= global_V) {
                    cerr << "[Rank " << rank << "] Error: Vertex " << vertex << " exceeds graph size " << global_V << endl << flush;
                    local_valid = false;
                    break;
                }
                float dist = (distStr == "INF") ? INF : stof(distStr);
                refDistances[vertex] = dist;
                refParents[vertex] = parent;
            }
            refFile.close();

            for (size_t i = 0; i < local_to_global.size(); ++i) {
                int global_v = local_to_global[i];
                bool syncDistMatch = (syncTree.Dist[i] == INF && refDistances[global_v] == INF) ||
                                     (abs(syncTree.Dist[i] - refDistances[global_v]) < EPSILON);
                bool syncParentMatch = syncTree.Parent[i] == refParents[global_v] ||
                                      (syncTree.Parent[i] != -1 && local_to_global[syncTree.Parent[i]] == refParents[global_v]);
                if (!syncDistMatch || !syncParentMatch) {
                    local_valid = false;
                    string error = "[Rank " + to_string(rank) + "] Sync mismatch for vertex " + to_string(global_v) +
                                   ": expected dist=" + (refDistances[global_v] == INF ? "INF" : to_string(refDistances[global_v])) +
                                   ", got dist=" + (syncTree.Dist[i] == INF ? "INF" : to_string(syncTree.Dist[i])) +
                                   "; expected parent=" + to_string(refParents[global_v]) +
                                   ", got parent=" + (syncTree.Parent[i] == -1 ? "-1" : to_string(local_to_global[syncTree.Parent[i]]));
                    local_sync_errors.push_back(error);
                }

                bool asyncDistMatch = (asyncTree.Dist[i] == INF && refDistances[global_v] == INF) ||
                                      (abs(asyncTree.Dist[i] - refDistances[global_v]) < EPSILON);
                bool asyncParentMatch = asyncTree.Parent[i] == refParents[global_v] ||
                                       (asyncTree.Parent[i] != -1 && local_to_global[asyncTree.Parent[i]] == refParents[global_v]);
                if (!asyncDistMatch || !asyncParentMatch) {
                    local_valid = false;
                    string error = "[Rank " + to_string(rank) + "] Async mismatch for vertex " + to_string(global_v) +
                                   ": expected dist=" + (refDistances[global_v] == INF ? "INF" : to_string(refDistances[global_v])) +
                                   ", got dist=" + (asyncTree.Dist[i] == INF ? "INF" : to_string(asyncTree.Dist[i])) +
                                   "; expected parent=" + to_string(refParents[global_v]) +
                                   ", got parent=" + (asyncTree.Parent[i] == -1 ? "-1" : to_string(local_to_global[asyncTree.Parent[i]]));
                    local_async_errors.push_back(error);
                }
            }
        }
    }

    if (local_valid) {
        cout << "[Rank " << rank << "] Local synchronous and asynchronous outputs are valid" << endl << flush;
    } else {
        if (!local_sync_errors.empty()) {
            cout << "[Rank " << rank << "] Local synchronous output has errors:" << endl << flush;
            for (const auto& error : local_sync_errors) {
                cout << error << endl << flush;
            }
        }
        if (!local_async_errors.empty()) {
            cout << "[Rank " << rank << "] Local asynchronous output has errors:" << endl << flush;
            for (const auto& error : local_async_errors) {
                cout << error << endl << flush;
            }
        }
    }

    bool sync_valid = true, async_valid = true;
    vector<string> sync_errors, async_errors;
    if (rank == 0) {
        vector<float> refDistances(global_V, INF);
        vector<int> refParents(global_V, -1);
        {
            ifstream refFile(referenceFile);
            if (!refFile.is_open()) {
                cerr << "[Rank " << rank << "] Error: Could not open reference file " << referenceFile << endl << flush;
                sync_valid = async_valid = false;
            } else {
                string line;
                getline(refFile, line);
                while (getline(refFile, line)) {
                    istringstream iss(line);
                    int vertex;
                    string distStr;
                    int parent;
                    if (!(iss >> vertex >> distStr >> parent)) {
                        cerr << "[Rank " << rank << "] Error: Invalid format in reference file at line: " << line << endl << flush;
                        sync_valid = async_valid = false;
                        break;
                    }
                    if (vertex >= global_V) {
                        cerr << "[Rank " << rank << "] Error: Vertex " << vertex << " exceeds graph size " << global_V << endl << flush;
                        sync_valid = async_valid = false;
                        break;
                    }
                    float dist = (distStr == "INF") ? INF : stof(distStr);
                    refDistances[vertex] = dist;
                    refParents[vertex] = parent;
                }
                refFile.close();
            }
        }

        {
            ifstream syncFile("combined_output_sync.txt");
            if (!syncFile.is_open()) {
                cerr << "[Rank " << rank << "] Error: Could not open combined_output_sync.txt" << endl << flush;
                sync_valid = false;
            } else {
                string line;
                getline(syncFile, line);
                int vertex_count = 0;
                while (getline(syncFile, line)) {
                    istringstream iss(line);
                    int vertex;
                    string distStr;
                    int parent;
                    if (!(iss >> vertex >> distStr >> parent)) {
                        cerr << "[Rank " << rank << "] Error: Invalid format in combined_output_sync.txt at line: " << line << endl << flush;
                        sync_valid = false;
                        break;
                    }
                    if (vertex >= global_V) {
                        cerr << "[Rank " << rank << "] Error: Vertex " << vertex << " exceeds graph size " << global_V << endl << flush;
                        sync_valid = false;
                        break;
                    }
                    float dist = (distStr == "INF") ? INF : stof(distStr);
                    bool distMatch = (dist == INF && refDistances[vertex] == INF) ||
                                     (abs(dist - refDistances[vertex]) < EPSILON);
                    bool parentMatch = parent == refParents[vertex];
                    if (!distMatch || !parentMatch) {
                        sync_valid = false;
                        string error = "[Rank " + to_string(rank) + "] Sync combined mismatch for vertex " + to_string(vertex) +
                                       ": expected dist=" + (refDistances[vertex] == INF ? "INF" : to_string(refDistances[vertex])) +
                                       ", got dist=" + (dist == INF ? "INF" : to_string(dist)) +
                                       "; expected parent=" + to_string(refParents[vertex]) +
                                       ", got parent=" + to_string(parent);
                        sync_errors.push_back(error);
                    }
                    vertex_count++;
                }
                if (vertex_count != global_V) {
                    cerr << "[Rank " << rank << "] Error: combined_output_sync.txt has " << vertex_count << " vertices, expected " << global_V << endl << flush;
                    sync_valid = false;
                }
                syncFile.close();
            }
        }

        {
            ifstream asyncFile("combined_output_async.txt");
            if (!asyncFile.is_open()) {
                cerr << "[Rank " << rank << "] Error: Could not open combined_output_async.txt" << endl << flush;
                async_valid = false;
            } else {
                string line;
                getline(asyncFile, line);
                int vertex_count = 0;
                while (getline(asyncFile, line)) {
                    istringstream iss(line);
                    int vertex;
                    string distStr;
                    int parent;
                    if (!(iss >> vertex >> distStr >> parent)) {
                        cerr << "[Rank " << rank << "] Error: Invalid format in combined_output_async.txt at line: " << line << endl << flush;
                        async_valid = false;
                        break;
                    }
                    if (vertex >= global_V) {
                        cerr << "[Rank " << rank << "] Error: Vertex " << vertex << " exceeds graph size " << global_V << endl << flush;
                        async_valid = false;
                        break;
                    }
                    float dist = (distStr == "INF") ? INF : stof(distStr);
                    bool distMatch = (dist == INF && refDistances[vertex] == INF) ||
                                     (abs(dist - refDistances[vertex]) < EPSILON);
                    bool parentMatch = parent == refParents[vertex];
                    if (!distMatch || !parentMatch) {
                        async_valid = false;
                        string error = "[Rank " + to_string(rank) + "] Async combined mismatch for vertex " + to_string(vertex) +
                                       ": expected dist=" + (refDistances[vertex] == INF ? "INF" : to_string(refDistances[vertex])) +
                                       ", got dist=" + (dist == INF ? "INF" : to_string(dist)) +
                                       "; expected parent=" + to_string(refParents[vertex]) +
                                       ", got parent=" + to_string(parent);
                        async_errors.push_back(error);
                    }
                    vertex_count++;
                }
                if (vertex_count != global_V) {
                    cerr << "[Rank " << rank << "] Error: combined_output_async.txt has " << vertex_count << " vertices, expected " << global_V << endl << flush;
                    async_valid = false;
                }
                asyncFile.close();
            }
        }

        if (sync_valid) {
            cout << "[Rank " << rank << "] Combined synchronous output is valid" << endl << flush;
        } else {
            cout << "[Rank " << rank << "] Combined synchronous output has errors:" << endl << flush;
            for (const auto& error : sync_errors) {
                cout << error << endl << flush;
            }
        }
        if (async_valid) {
            cout << "[Rank " << rank << "] Combined asynchronous output is valid" << endl << flush;
        } else {
            cout << "[Rank " << rank << "] Combined asynchronous output has errors:" << endl << flush;
            for (const auto& error : async_errors) {
                cout << error << endl << flush;
            }
        }

        if (sync_valid && async_valid) {
            cout << "[Rank " << rank << "] All outputs are correct" << endl << flush;
        } else {
            cout << "[Rank " << rank << "] Errors found in " << (sync_valid ? "" : "synchronous ") << (async_valid ? "" : "asynchronous ") << "output" << endl << flush;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return local_valid && sync_valid && async_valid;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "[Rank " << rank << "] Starting program with " << size << " processes" << endl << flush;

    if (argc != 4) {
        if (rank == 0) {
            cerr << "[Rank " << rank << "] Usage: " << argv[0] << " <graph_file> <changes_file> <source_vertex>" << endl << flush;
        }
        MPI_Finalize();
        return 1;
    }

    PerformanceMetrics metrics;
    metrics.mpi_rank = rank;
    double program_start = omp_get_wtime();
    metrics.num_threads = omp_get_max_threads();

    string graphFile = argv[1];
    string changesFile = argv[2];
    int global_source = stoi(argv[3]);
    int async_level = 3;

    cout << "[Rank " << rank << "] Input: graph=" << graphFile << ", changes=" << changesFile << ", source=" << global_source << endl << flush;

    Graph global_g;
    vector<Edge> changes;
    if (rank == 0) {
        global_g = readGraph(graphFile, metrics);
        changes = readChanges(changesFile, metrics);
    }

    int global_V = rank == 0 ? global_g.V : 0;
    MPI_Bcast(&global_V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    global_g.V = global_V;
    if (rank != 0) {
        global_g.adj.resize(global_V);
    }

    for (int v = 0; v < global_V; ++v) {
        int adj_size = rank == 0 ? global_g.adj[v].size() : 0;
        MPI_Bcast(&adj_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            global_g.adj[v].resize(adj_size);
        }
        MPI_Bcast(global_g.adj[v].data(), adj_size * sizeof(pair<int, float>), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    cout << "[Rank " << rank << "] Global graph size: " << global_V << " vertices" << endl << flush;

    int num_changes = rank == 0 ? changes.size() : 0;
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        changes.resize(num_changes);
    }
    MPI_Bcast(changes.data(), num_changes * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);
    cout << "[Rank " << rank << "] Received " << num_changes << " changes" << endl << flush;

    vector<int> part(global_V, 0);
    if (rank == 0) {
        partitionGraphWithMetis(global_g, size, part, metrics);
    }
    MPI_Bcast(part.data(), global_V, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> local_to_global;
    vector<int> global_to_local(global_V, -1);
    Graph local_g = buildLocalGraph(global_g, part, rank, local_to_global);

    for (size_t i = 0; i < local_to_global.size(); ++i) {
        global_to_local[local_to_global[i]] = i;
    }

    vector<int> boundary_vertices;
    vector<vector<int>> vertex_to_procs(global_V);
    identifyBoundaryVertices(global_g, part, rank, boundary_vertices, vertex_to_procs);

    SSSPTree syncTree, asyncTree;
    int local_source = global_to_local[global_source] != -1 ? global_to_local[global_source] : -1;
    if (local_source != -1) {
        syncTree = initSSSP(local_g, local_source, metrics);
    } else {
        syncTree.Parent.resize(local_g.V, -1);
        syncTree.Dist.resize(local_g.V, INF);
        syncTree.Affected_Del.resize(local_g.V, false);
        syncTree.Affected.resize(local_g.V, false);
        syncTree.Children.resize(local_g.V);
        cout << "[Rank " << rank << "] Source vertex not in partition, initializing empty SSSP tree" << endl << flush;
    }
    asyncTree = syncTree;

    identifyAffectedVertices(local_g, syncTree, changes, metrics, global_to_local);
    identifyAffectedVertices(local_g, asyncTree, changes, metrics, global_to_local);

    bool global_changed;
    int sync_iter = 0;
    do {
        sync_iter++;
        cout << "[Rank " << rank << "] Synchronous processing iteration " << sync_iter << endl << flush;
        updateAffectedVertices(local_g, syncTree, metrics);
        communicateBoundaryUpdates(syncTree, boundary_vertices, vertex_to_procs,
                                 local_to_global, global_to_local, metrics);
        int local_changed = anyTrue(syncTree.Affected) ? 1 : 0;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        cout << "[Rank " << rank << "] Synchronous global_changed: " << global_changed << endl << flush;
    } while (global_changed);

    double sync_update_time = metrics.update_time;

    metrics.update_time = 0;
    int async_iter = 0;
    do {
        async_iter++;
        cout << "[Rank " << rank << "] Asynchronous processing iteration " << async_iter << endl << flush;
        asyncUpdateAffectedVertices(local_g, asyncTree, async_level, metrics);
        communicateBoundaryUpdates(asyncTree, boundary_vertices, vertex_to_procs,
                                 local_to_global, global_to_local, metrics);
        int local_changed = anyTrue(asyncTree.Affected) ? 1 : 0;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        cout << "[Rank " << rank << "] Asynchronous global_changed: " << global_changed << endl << flush;
    } while (global_changed);

    double async_update_time = metrics.update_time;

    double program_end = omp_get_wtime();
    metrics.total_time = program_end - program_start;
    double serial_time = 0.290735;
    metrics.speedup = serial_time / metrics.total_time;
    metrics.update_method = "both";

    saveResults(syncTree, metrics, local_to_global, "_sync");
    saveResults(asyncTree, metrics, local_to_global, "_async");

    combineOutputs(syncTree, asyncTree, local_to_global, global_V);

    compareWithReferenceOutput(syncTree, asyncTree, local_to_global, global_V, "output.txt");

    metrics.print();

    cout << "[Rank " << rank << "] Program completed" << endl << flush;
    MPI_Finalize();
    return 0;
}