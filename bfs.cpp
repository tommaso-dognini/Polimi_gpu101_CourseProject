#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>

#define MAX_FRONTIER_SIZE 128

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals);

void insertIntoFrontier(int val, int *frontier, int *frontier_size)
{
  frontier[*frontier_size] = val;
  *frontier_size = *frontier_size + 1;
}

inline void swap(int **ptr1, int **ptr2)
{
  int *tmp = *ptr1;
  *ptr1 = *ptr2;
  *ptr2 = tmp;
}

void BFS_sequential(const int source, const int *rowPointers, const int *destinations, int *distances)
{
  int **frontiers = (int **)malloc(2 * sizeof(int *));
  for (int i = 0; i < 2; i++)
    frontiers[i] = (int *)calloc(MAX_FRONTIER_SIZE, sizeof(int));
  int *currentFrontier = frontiers[0];
  int currentFrontierSize = 0;
  int *previousFrontier = frontiers[1];
  int previousFrontierSize = 0;
  insertIntoFrontier(source, previousFrontier, &previousFrontierSize);
  distances[source] = 0;
  while (previousFrontierSize > 0)
  {
    // visit all vertices on the previous frontier
    for (int f = 0; f < previousFrontierSize; f++)
    {
      const int currentVertex = previousFrontier[f];
      // check all outgoing edges
      for (int i = rowPointers[currentVertex]; i < rowPointers[currentVertex + 1]; ++i)
      {
        if (distances[destinations[i]] == -1)
        {
          // this vertex has not been visited yet
          insertIntoFrontier(destinations[i], currentFrontier, &currentFrontierSize);
          distances[destinations[i]] = distances[currentVertex] + 1;
        }
      }
    }
    swap(&currentFrontier, &previousFrontier);
    previousFrontierSize = currentFrontierSize;
    currentFrontierSize = 0;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: ./exec matrix_file source\n");
    return 0;
  }

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<float> values;
  int num_rows, num_cols, num_vals;

  const std::string filename = argv[1];
  // The node starts from 1 but array starts from 0
  const int source = atoi(argv[2]) - 1;

  read_matrix(row_ptr, col_ind, values, filename, num_rows, num_cols, num_vals);

  // Initialize dist to -1
  std::vector<int> dist(num_vals);
  for (int i = 0; i < num_vals; i++)
  {
    dist[i] = -1;
  }
  // Compute in sw
  BFS_sequential(source, row_ptr.data(), col_ind.data(), dist.data());

  //Visualize the result
  for (int i = 0; i < dist.size(); i++)
  {
    std::cout << "Nodo " << i + 1 << ": Distanza = " << dist[i] << std::endl;
  }
  return EXIT_SUCCESS;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "File cannot be opened!\n";
    throw std::runtime_error("File cannot be opened");
  }

  // Get number of rows, columns, and non-zero values
  file >> num_rows >> num_cols >> num_vals;

  row_ptr.resize(num_rows + 1);
  col_ind.resize(num_vals);
  values.resize(num_vals);

  // Collect occurrences of each row for determining the indices of row_ptr
  std::vector<int> row_occurrences(num_rows, 0);

  int row, column;
  float value;
  while (file >> row >> column >> value)
  {
    // Subtract 1 from row and column indices to match C format
    row--;
    column--;

    row_occurrences[row]++;
  }

  // Set row_ptr
  int index = 0;
  for (int i = 0; i < num_rows; i++)
  {
    row_ptr[i] = index;
    index += row_occurrences[i];
  }
  row_ptr[num_rows] = num_vals;

  // Reset the file stream to read again from the beginning
  file.clear();
  file.seekg(0, std::ios::beg);

  // Read the first line again to skip it
  file >> num_rows >> num_cols >> num_vals;

  std::fill(col_ind.begin(), col_ind.end(), -1);

  int i = 0;
  while (file >> row >> column >> value)
  {
    row--;
    column--;

    // Find the correct index (i + row_ptr[row]) using both row information and an index i
    while (col_ind[i + row_ptr[row]] != -1)
    {
      i++;
    }
    col_ind[i + row_ptr[row]] = column;
    values[i + row_ptr[row]] = value;
    i = 0;
  }
}
