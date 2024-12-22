bfs: bfs.cpp
	g++ -g3 -Wall -Werror bfs.cpp -o bfs

gpu_bfs: gpu_bfs.cu
	nvcc -Wall -Werror gpu_bfs.cu -o gpu_bfs

clean:
	rm -f bfs
	rm -f gpu_bfs