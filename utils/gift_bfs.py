import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os

# Create the graph
G = nx.Graph()
edges = [(0, 1), (0, 3), (1, 2), (3, 4), (4, 5), (2, 5)]
G.add_edges_from(edges)

# Positions for nodes
pos = nx.circular_layout(G)

# BFS simulation
def bfs_animation(graph, start_node):
    visited = set()
    queue = [start_node]
    frames = []

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)

            # Highlight the current node and its neighbors
            node_colors = ['red' if node == current else 
                           'green' if node in visited else 
                           'lightgray' 
                           for node in graph.nodes()]
            edge_colors = ['black' if (current in edge and edge[1] in visited) or 
                                    (current in edge and edge[0] in visited) else 
                           'gray' 
                           for edge in graph.edges()]

            # Draw the graph
            plt.figure(figsize=(6, 6))
            nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=800, font_weight='bold')
            plt.title(f"Exploring Node {current}", fontsize=14)
            frame_path = f"frame_{len(frames)}.png"
            plt.savefig(frame_path)
            plt.close()
            frames.append(frame_path)

            # Add unvisited neighbors to the queue
            for neighbor in graph.neighbors(current):
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)

    # Add additional frames for a 10-second pause
    for _ in range(10):  # Assuming 1 second per frame
        frames.append(frames[-1])  # Duplicate the last frame

    return frames

# Generate frames for BFS starting from node 0
frames = bfs_animation(G, start_node=0)

# Create a GIF with a loop and 10-second pause
with imageio.get_writer('bfs_animation.gif', mode='I', duration=1) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Cleanup: Remove individual frames
for frame in frames[:-10]:  # Exclude pause frames
    os.remove(frame)

print("GIF created: bfs_animation.gif")