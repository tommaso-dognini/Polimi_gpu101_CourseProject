import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread
import io


filename = "matrix_test.mtx"  # file senza banner
with open(filename, 'r') as f:
    content = f.read()

# Se la riga banner manca, la aggiungiamo al volo
if not content.startswith("%%MatrixMarket"):
    banner = "%%MatrixMarket matrix coordinate real general\n"
    content = banner + content

# Ora usiamo mmread su uno stream (senza creare un file fisico)
# Leggi la matrice dal file .mtx (Matrix Market)
# Sostituisci con il tuo file
M = mmread(io.StringIO(content))

# Converti in formato COOrdinate per ottenere liste row, col, data
M_coo = M.tocoo()

# Costruisci un grafo diretto (usa nx.Graph() se il grafo è non diretto)
G = nx.DiGraph()

# Aggiungi archi per ogni valore != 0
# ATTENZIONE: Se la matrice .mtx è 1-based, potresti dover fare (row-1, col-1)
for row, col, val in zip(M_coo.row, M_coo.col, M_coo.data):
    # Se val == 0, ignora
    if val != 0:
        G.add_edge(row, col)

# Visualizza a schermo con Matplotlib
plt.figure(figsize=(8,8))
pos = nx.spring_layout(G)  # Layout “a molla”, uno dei tanti disponibili
nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
plt.show()