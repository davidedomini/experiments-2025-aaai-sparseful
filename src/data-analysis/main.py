import pickle
import pandas as pd
import seaborn as sns
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster

def load_adjacency_matrix(file_path):
    with open(file_path, 'rb') as f:
        adj_matrix = pickle.load(f, encoding='latin1')
    adjacency = adj_matrix[2]
    return adjacency

def plot_correlation_matirx(data, chart_path):
    corr_matrix = data.corr(method="pearson")  # 207 x 207
    linkage_matrix = linkage(corr_matrix, method="ward")
    dendro_order = leaves_list(linkage_matrix)
    corr_matrix_ordered = corr_matrix.iloc[dendro_order, :].iloc[:, dendro_order]

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        corr_matrix_ordered,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson Correlation"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Corellation among sensors heatmap (odered by similarity)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{chart_path}correlation_matrix.pdf')
    plt.close()


def plot_clustered_nodes(data, adjacency, n_clusters, chart_path):
    corr = data.corr()
    link = linkage(corr, method="ward")
    cluster_labels = fcluster(link, t=n_clusters, criterion="maxclust")
    sensor_to_cluster = dict(zip(data.columns, cluster_labels))

    # Remove self-loops
    G = nx.from_numpy_array(adjacency)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Map nodes to clusters
    node_indices = list(G.nodes())
    node_clusters = [sensor_to_cluster[data.columns[n]] for n in node_indices]
    unique_clusters = sorted(set(cluster_labels))
    palette = sns.color_palette("tab10", len(unique_clusters))

    # Group nodes by cluster
    clusters = []
    for c in unique_clusters:
        cluster_nodes = [n for n, cl in zip(node_indices, node_clusters) if cl == c]
        clusters.append(cluster_nodes)

    # Position clusters in a circular layout
    superG = nx.cycle_graph(len(clusters))
    superpos = nx.spring_layout(superG, scale=5, seed=42)

    # Final positions
    pos = dict()
    for i, nodes in enumerate(clusters):
        # Local layout of cluster i
        subG = G.subgraph(nodes)
        subpos = nx.spring_layout(subG, scale=1, seed=42)

        # Global center of cluster i
        cluster_center = superpos[i]

        # Position each node of cluster i around the global center
        for node, p in subpos.items():
            pos[node] = p + cluster_center

    # Color nodes by cluster
    node_colors = []
    for cl in node_clusters:
        node_colors.append(palette[cl - 1])

   
    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.4, alpha=0.5)
    plt.axis("off")
    plt.savefig(f'{chart_path}sensor_clusters.pdf')



if __name__ == '__main__':
    data_path = 'data/METR-LA/'
    chart_path = 'charts/METR-LA/'

    Path(chart_path).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_hdf(f'{data_path}METR-LA.h5')
    adjacency = load_adjacency_matrix(f'{data_path}adj_METR-LA.pkl')

    # Charts
    plot_correlation_matirx(df, chart_path)
    plot_clustered_nodes(df, adjacency, 6, chart_path)