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

def plot_correlation_matrix(data, chart_path):
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


import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster


def select_top_correlated_sensors_split(
        data: pd.DataFrame,
        n_clusters: int = 6,
        n_per_group: int = 5
):
    # Compute correlation matrix
    corr = data.corr()

    # Hierarchical clustering
    link = linkage(corr, method="ward")
    cluster_labels = fcluster(link, t=n_clusters, criterion="maxclust")

    # Map sensors to clusters
    sensor_to_cluster = dict(zip(data.columns, cluster_labels))

    # Initialize mappings and selected columns
    train_map = dict()
    val_map = dict()
    test_map = dict()

    train_columns = []
    val_columns = []
    test_columns = []

    # For each cluster
    for cl in sorted(set(cluster_labels)):
        # Sensors in the cluster
        cluster_cols = [col for col, lab in sensor_to_cluster.items() if lab == cl]

        # Sub-correlation matrix
        sub_corr = corr.loc[cluster_cols, cluster_cols].copy()

        # Average correlation per sensor
        avg_corr = sub_corr.mean(axis=1)

        # Sort sensors descending by avg correlation
        sorted_sensors = avg_corr.sort_values(ascending=False).index.tolist()

        # Select N sensors per group
        top_train = sorted_sensors[:n_per_group]
        top_val = sorted_sensors[n_per_group:n_per_group * 2]
        top_test = sorted_sensors[n_per_group * 2:n_per_group * 3]

        # Store mapping
        train_map[cl] = top_train
        val_map[cl] = top_val
        test_map[cl] = top_test

        # Append to global lists
        train_columns.extend(top_train)
        val_columns.extend(top_val)
        test_columns.extend(top_test)

    # Subset data
    train_data = data[train_columns]
    val_data = data[val_columns]
    test_data = data[test_columns]

    return train_data, val_data, test_data, train_map, val_map, test_map


def save_clusters(clusters_map, split, data_path):
    sorted_clusters = sorted(clusters_map.keys())
    col_data = {}
    for cl in sorted_clusters:
        col_name = f"cluster-{cl}"
        col_data[col_name] = clusters_map[cl]
    df = pd.DataFrame.from_dict(col_data, orient='columns')
    
    df.to_csv(f'{data_path}clusters-{split}.csv', index=False)


if __name__ == '__main__':
    data_path = 'data/METR-LA/'
    chart_path = 'charts/METR-LA/'

    Path(chart_path).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_hdf(f'{data_path}METR-LA.h5')
    adjacency = load_adjacency_matrix(f'{data_path}adj_METR-LA.pkl')

    # Charts
    plot_correlation_matrix(df, chart_path)
    plot_clustered_nodes(df, adjacency, 6, chart_path)

    filtered_df_train, filtered_df_val, filtered_df_test, clusters_map_train, clusters_map_val, clusters_map_test = select_top_correlated_sensors_split(df)
    filtered_df_train.to_csv(f'{data_path}reduced_METR-LA-train.csv', index=False)
    filtered_df_val.to_csv(f'{data_path}reduced_METR-LA-val.csv', index=False)
    filtered_df_test.to_csv(f'{data_path}reduced_METR-LA-test.csv', index=False)
    save_clusters(clusters_map_train, 'train', data_path)
    save_clusters(clusters_map_val, 'val', data_path)
    save_clusters(clusters_map_test, 'test', data_path)