#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bfs_kernel(int *d_graph,
                           int *d_edges,
                           int *d_frontier,
                           int *d_next,
                           int *d_visited,
                           int *d_active,
                           int num_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Se tid è un nodo valido e appartiene alla frontiera corrente
    if (tid < num_nodes && d_frontier[tid] == 1) {
        // Il nodo tid è processato in questo livello, quindi lo togliamo dalla frontiera
        d_frontier[tid] = 0;

        // Indichiamo che almeno un nodo è stato elaborato in questo livello
        // (basta una OR per impostare d_active a 1, se non lo era già)
        atomicOr(d_active, 1);

        // Scorriamo i vicini del nodo tid
        int start = d_graph[tid];
        int end   = d_graph[tid + 1];
        for (int i = start; i < end; i++) {
            int neighbor = d_edges[i];
            // Se il vicino non è ancora stato visitato
            if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                // Aggiungilo ai nodi attivi del prossimo livello
                d_next[neighbor] = 1;
            }
        }
    }
}

void bfs(int *h_graph, int *h_edges, int start_node, int num_nodes, int num_edges)
{
    // Variabili su device
    int *d_graph, *d_edges, *d_frontier, *d_next, *d_visited, *d_active;

    // Array su host per inizializzare frontier e visited
    int *h_frontier = (int*)calloc(num_nodes, sizeof(int));
    int *h_visited  = (int*)calloc(num_nodes, sizeof(int));

    // Inizializziamo la frontiera col solo nodo di partenza
    h_frontier[start_node] = 1;
    h_visited[start_node]  = 1;

    // Allocazione su device
    cudaMalloc(&d_graph,    (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_edges,    num_edges      * sizeof(int));
    cudaMalloc(&d_frontier, num_nodes      * sizeof(int));
    cudaMalloc(&d_next,     num_nodes      * sizeof(int));
    cudaMalloc(&d_visited,  num_nodes      * sizeof(int));
    cudaMalloc(&d_active,   sizeof(int));

    // Copiamo i dati dal host al device
    cudaMemcpy(d_graph,    h_graph,    (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges,    h_edges,    num_edges      * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, h_frontier, num_nodes      * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited,  h_visited,  num_nodes      * sizeof(int),  cudaMemcpyHostToDevice);

    // Azzeriamo d_next, che conterrà la frontiera del livello successivo
    cudaMemset(d_next, 0, num_nodes * sizeof(int));

    // Parametri di esecuzione: ad esempio 256 thread per blocco
    dim3 blockSize(256);
    dim3 gridSize((num_nodes + blockSize.x - 1) / blockSize.x);

    while (true) {
        // Azzeriamo d_active prima di lanciare il kernel
        cudaMemset(d_active, 0, sizeof(int));

        // Lanciamo il kernel BFS
        bfs_kernel<<<gridSize, blockSize>>>(d_graph, d_edges,
                                            d_frontier, d_next, d_visited,
                                            d_active, num_nodes);

        // Sincronizziamo per assicurarci che il kernel sia terminato
        cudaDeviceSynchronize();

        // Copiamo d_active su host per capire se si sono attivati nuovi nodi
        int active = 0;
        cudaMemcpy(&active, d_active, sizeof(int), cudaMemcpyDeviceToHost);

        // Se active è 0, non ci sono stati nuovi nodi attivi in questo livello:
        // la BFS è terminata
        if (active == 0) {
            break;
        }

        // Altrimenti spostiamo i nodi del prossimo livello in frontier
        cudaMemcpy(d_frontier, d_next, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice);

        // E azzeriamo d_next in vista del prossimo giro
        cudaMemset(d_next, 0, num_nodes * sizeof(int));
    }

    // Clean up
    cudaFree(d_graph);
    cudaFree(d_edges);
    cudaFree(d_frontier);
    cudaFree(d_next);
    cudaFree(d_visited);
    cudaFree(d_active);
    free(h_frontier);
    free(h_visited);
}

int main()
{
    // Esempio: Graph data
    // h_graph contiene gli offset CSR, h_edges i vicini
    // qui aggiungi i tuoi dati concreti...
    
    int num_nodes = 6;
    int num_edges = 7;
    int start_node = 0;

    // Creiamo un semplice esempio di h_graph, h_edges
    // in modo che la BFS parta dal nodo 0.

    int *h_graph = (int*)malloc((num_nodes + 1) * sizeof(int));
    int *h_edges = (int*)malloc(num_edges * sizeof(int));

    // Esempio fittizio, da rimpiazzare con dati reali
    // h_graph[i]  = indice dell'inizio della lista di adiacenza di i in h_edges
    // h_graph[i+1]= indice della fine della lista di adiacenza di i in h_edges
    // ...
    
    // Inizializzazione a scopo dimostrativo
    h_graph[0] = 0;  // offset per nodo 0
    h_graph[1] = 1;  // offset per nodo 1
    h_graph[2] = 2;  
    h_graph[3] = 3;  
    h_graph[4] = 4;  
    h_graph[5] = 5;  
    h_graph[6] = 7;  // l'ultimo è la fine dell'array h_edges

    // h_edges: adiacenze vere e proprie
    // (questo è giusto un esempio)
    h_edges[0] = 1;
    h_edges[1] = 2;
    h_edges[2] = 3;
    h_edges[3] = 4;
    h_edges[4] = 5;
    h_edges[5] = 0;
    h_edges[6] = 1;

    // Lancia la BFS
    bfs(h_graph, h_edges, start_node, num_nodes, num_edges);

    // Liberiamo
    free(h_graph);
    free(h_edges);
    return 0;
}