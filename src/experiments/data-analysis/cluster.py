import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import h5py
import os

# --- Configurazione dei percorsi dei file ---
# Assicurati che questo file sia nella stessa directory dello script
# o aggiorna il percorso di conseguenza.
DATA_DIR = 'data/METR-LA/' # La directory dove si trova il file del dataset
TRAFFIC_DATA_FILE = os.path.join(DATA_DIR, 'METR-LA.h5')
OUTPUT_CSV_FILE = 'metr_la_top_correlated_sensor_clusters.csv' # Nome del file CSV di output

# --- Funzione per caricare i dati di traffico (usando pd.read_hdf) ---
def load_traffic_data(file_path):
    """
    Carica i dati di velocità del traffico dal file.h5 usando pd.read_hdf.
    """
    try:
        # pd.read_hdf legge direttamente il DataFrame
        traffic_df = pd.read_hdf(file_path)
        # Assicurati che gli ID dei sensori siano stringhe per coerenza
        traffic_df.columns = traffic_df.columns.astype(str)
        print(f"Caricati dati di traffico con forma: {traffic_df.shape} da {file_path} usando pd.read_hdf.")
        return traffic_df
    except FileNotFoundError:
        print(f"Errore: File non trovato a {file_path}. Assicurati di aver scaricato il dataset.")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento dei dati di traffico: {e}")
        return None

# --- Funzione per estrarre le feature e pulire i dati ---
def extract_and_clean_features(traffic_df):
    """
    Estrae le feature statistiche dai dati di traffico e gestisce i valori mancanti.
    I valori 0 sono trattati come mancanti, poiché la ricerca indica che rappresentano dati assenti.
    """
    if traffic_df is None or traffic_df.empty:
        print("DataFrame dei dati di traffico vuoto o non caricato. Impossibile estrarre le feature.")
        return None

    # Sostituisci i valori 0 con NaN, poiché 0 indica dati mancanti in questo dataset [2, 6, 3, 4]
    # Questo assicura che le statistiche siano calcolate solo su valori di traffico reali.
    traffic_df_cleaned = traffic_df.replace(0, np.nan)

    print("\nEstrazione delle feature statistiche per ciascun sensore (ignorando i valori mancanti)...")
    features = pd.DataFrame({
        'mean_speed': traffic_df_cleaned.mean(axis=0),
        'std_speed': traffic_df_cleaned.std(axis=0),
        'median_speed': traffic_df_cleaned.median(axis=0),
        'min_speed': traffic_df_cleaned.min(axis=0),
        'max_speed': traffic_df_cleaned.max(axis=0),
        'q10_speed': traffic_df_cleaned.quantile(0.10, axis=0),
        'q90_speed': traffic_df_cleaned.quantile(0.90, axis=0),
        'missing_ratio': (traffic_df == 0).sum(axis=0) / len(traffic_df) # Calcola il rapporto di valori 0 (mancanti)
    })

    # Gestisci i sensori che potrebbero avere tutti i valori mancanti (NaN) dopo la pulizia
    # Questo può accadere se un sensore registra solo 0 o NaN per l'intero periodo.
    initial_sensors_count = len(features)
    features = features.dropna()
    if len(features) < initial_sensors_count:
        print(f"Attenzione: {initial_sensors_count - len(features)} sensori sono stati rimossi perché contenevano solo valori mancanti (0).")

    print(f"Feature estratte per {len(features)} sensori.")
    return features, traffic_df_cleaned # Restituisci anche il df pulito per la correlazione

# --- Funzione per eseguire il clustering dei sensori ---
def cluster_sensors(features_df, n_clusters=5, random_state=42):
    """
    Esegue il clustering K-Means sui sensori basandosi sulle feature dei dati di traffico.
    """
    if features_df is None or features_df.empty:
        print("DataFrame delle feature vuoto o non caricato. Impossibile eseguire il clustering.")
        return None

    # Normalizza le feature per garantire che nessuna feature domini il clustering [5]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    print(f"\nEsecuzione del clustering K-Means con {n_clusters} cluster sulle feature dei dati di traffico...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10) # n_init per evitare warning [5]
    features_df['cluster_label'] = kmeans.fit_predict(X_scaled)

    print("\nDivisione dei sensori in cluster (basata sui pattern di traffico):")
    clusters = {}
    for i in range(n_clusters):
        cluster_sensors_ids = features_df[features_df['cluster_label'] == i].index.tolist()
        clusters[f'Cluster {i+1}'] = cluster_sensors_ids
        print(f"Cluster {i+1} ({len(cluster_sensors_ids)} sensori): {cluster_sensors_ids}")
    
    return clusters, features_df

# --- Funzione per selezionare i sensori più correlati all'interno di ogni cluster ---
def get_top_correlated_sensors_per_cluster(traffic_data_cleaned_df, clusters_dict, n_top_correlated=10):
    """
    Per ogni cluster, seleziona i N_TOP_CORRELATED sensori più correlati tra loro.
    La correlazione è basata sulla serie temporale dei dati di traffico.
    """
    final_clusters_to_save = {}
    print(f"\nSelezione dei primi {n_top_correlated} sensori più correlati per ciascun cluster...")

    for cluster_name, sensor_ids_in_cluster in clusters_dict.items():
        if not sensor_ids_in_cluster:
            #final_clusters_to_save[cluster_name] =
            print(f"{cluster_name}: Nessun sensore valido per la correlazione.")
            continue

        # Filtra i dati di traffico per i sensori di questo cluster
        # Assicurati che i sensori siano presenti nel DataFrame pulito
        valid_sensors_in_cluster = [s_id for s_id in sensor_ids_in_cluster if s_id in traffic_data_cleaned_df.columns]
        
        if not valid_sensors_in_cluster:
            #final_clusters_to_save[cluster_name] =
            print(f"{cluster_name}: Nessun sensore valido nel DataFrame di traffico pulito per la correlazione.")
            continue

        cluster_data = traffic_data_cleaned_df[valid_sensors_in_cluster]

        # Calcola la matrice di correlazione. `min_periods=1` per gestire sensori con pochi dati non-NaN.
        # I valori NaN nella serie temporale saranno gestiti da.corr()
        correlation_matrix = cluster_data.corr(min_periods=1) 
        
        # Rimuovi i NaN che potrebbero derivare da sensori con troppi dati mancanti
        # (es. se un sensore ha solo NaN dopo la pulizia, la sua riga/colonna nella matrice di correlazione sarà NaN)
        correlation_matrix = correlation_matrix.dropna(axis=0).dropna(axis=1)

        if correlation_matrix.empty:
            #final_clusters_to_save[cluster_name] =
            print(f"{cluster_name}: Matrice di correlazione vuota dopo la pulizia. Nessun sensore selezionato.")
            continue

        # Calcola la somma delle correlazioni assolute per ogni sensore con gli altri nel cluster
        # Escludi la correlazione del sensore con se stesso (che è 1)
        sum_abs_correlations = {}
        for s_id in correlation_matrix.columns:
            # Somma le correlazioni assolute con tutti gli altri sensori nel cluster
            # Escludi la correlazione con se stesso (s_id == col)
            sum_abs_correlations[s_id] = correlation_matrix[s_id].drop(s_id, errors='ignore').abs().sum()
        
        # Ordina i sensori in base alla somma delle correlazioni assolute (dal più alto al più basso)
        sorted_sensors = sorted(sum_abs_correlations.items(), key=lambda item: item[1], reverse=True)
        
        # Seleziona i primi N_TOP_CORRELATED sensori
        top_sensors = [s_id for s_id, _ in sorted_sensors[:n_top_correlated]]
        
        final_clusters_to_save[cluster_name] = top_sensors
        print(f"{cluster_name}: Sensori più correlati selezionati ({len(top_sensors)}): {top_sensors}")
    
    return final_clusters_to_save

# --- Funzione per salvare i cluster in un CSV ---
def save_clusters_to_csv(clusters_dict, output_file):
    """
    Salva i cluster di sensori in un file CSV.
    Ogni colonna rappresenta un cluster, e le righe contengono gli ID dei sensori.
    """
    # Trova la lunghezza massima di un cluster per dimensionare il DataFrame
    max_len = max(len(v) for v in clusters_dict.values()) if clusters_dict else 0

    # Crea un dizionario dove ogni chiave è il nome del cluster e il valore è una lista di sensori
    # riempita con NaN per uniformare le lunghezze
    data_for_df = {
        col_name: (sensors_list + [np.nan] * (max_len - len(sensors_list)))
        for col_name, sensors_list in clusters_dict.items()
    }

    clusters_df = pd.DataFrame(data_for_df)
    clusters_df.to_csv(output_file, index=False)
    print(f"\nCluster dei sensori salvati in '{output_file}'")

# --- Esecuzione principale dello script ---
if __name__ == "__main__":
    print("Avvio dell'analisi del dataset METR-LA per il clustering dei sensori basato sui dati di traffico...")

    # 1. Carica i dati di traffico
    traffic_data_df = load_traffic_data(TRAFFIC_DATA_FILE)

    if traffic_data_df is not None:
        # 2. Estrai le feature e pulisci i dati (restituisce anche il df pulito per la correlazione)
        sensor_features_df, traffic_data_cleaned_for_corr = extract_and_clean_features(traffic_data_df)

        if sensor_features_df is not None:
            # 3. Esegui il clustering K-Means
            num_clusters = 5 # Puoi modificare il numero di cluster
            sensor_clusters, updated_features_df = cluster_sensors(sensor_features_df, n_clusters=num_clusters)

            # 4. Seleziona i sensori più correlati all'interno di ogni cluster
            n_top_correlated_sensors = 10 # Numero di sensori più correlati da selezionare per cluster
            top_correlated_clusters = get_top_correlated_sensors_per_cluster(
                traffic_data_cleaned_for_corr, sensor_clusters, n_top_correlated=n_top_correlated_sensors
            )

            # 5. Salva i cluster selezionati in un file CSV
            if top_correlated_clusters:
                save_clusters_to_csv(top_correlated_clusters, OUTPUT_CSV_FILE)
            else:
                print("Nessun cluster con sensori correlati da salvare.")

            print("\n--- Riepilogo delle proprietà medie per ciascun cluster (basato su tutti i sensori del cluster) ---")
            for cluster_name, sensor_ids in sensor_clusters.items(): # Usa i cluster originali per il riepilogo delle proprietà
                if sensor_ids:
                    # Assicurati di selezionare solo le colonne delle feature originali
                    cluster_avg_features = updated_features_df[updated_features_df.index.isin(sensor_ids)].drop(columns=['cluster_label']).mean()
                    print(f"\n{cluster_name}:")
                    print(cluster_avg_features)
                else:
                    print(f"\n{cluster_name}: Nessun sensore in questo cluster.")

    print("\nAnalisi completata.")