# src/hcbou_impl.py

import pandas as pd
import numpy as np
import warnings
from collections import Counter
from typing import Tuple, Dict, Optional, Any

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def roc_auc_score_multiclass(y_test, y_pred, average="macro"):

    # Creating a set of all the unique classes using the actual class list
    unique_class = set(y_test)
    roc_auc_dict = {}

    for per_class in unique_class:
        # Creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # Marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in y_test]
        new_pred_class = [0 if x in other_class else 1 for x in y_pred]

        # Using the sklearn metrics method to calculate the roc_auc_score
        try:
            roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
            roc_auc_dict[per_class] = roc_auc
        except ValueError:
            # Handle cases where only one class is present
            roc_auc_dict[per_class] = 0.5

    if average == "macro":
        return sum(roc_auc_dict.values()) / len(roc_auc_dict.values())
    else:
        return roc_auc_dict


def majority_class_undersampling(X_majority, y_majority, target_size, max_clusters=8, random_state=42):

    # print(f"[Mayoritaria] Aplicando submuestreo")

    # Create temporary dataset for ClusterCentroids
    temp_majority = X_majority.copy()
    majority_class_label = y_majority.iloc[0]  # Get the actual label

    # Convert labels to strings to avoid type conflicts with LabelEncoder
    majority_class_str = str(majority_class_label)
    fake_minority_str = 'fake_minority'

    temp_majority['class'] = majority_class_str

    # Add fake minority samples to enable ClusterCentroids
    fake_minority = X_majority.sample(n=target_size, replace=True, random_state=random_state).copy()
    fake_minority['class'] = fake_minority_str

    # Combine datasets
    temp_data = pd.concat([temp_majority, fake_minority])
    X_temp = temp_data.drop('class', axis=1)
    y_temp = temp_data['class']

    # Convert to numeric labels for ClusterCentroids
    le = LabelEncoder()
    y_temp_encoded = le.fit_transform(y_temp)

    # Apply ClusterCentroids
    cc = ClusterCentroids(
        estimator=MiniBatchKMeans(n_clusters=max_clusters, n_init=1, random_state=random_state),
        random_state=random_state
    )

    X_resampled, y_resampled = cc.fit_resample(X_temp, y_temp_encoded)

    # Extract only the majority class samples
    majority_mask = y_resampled == le.transform([majority_class_str])[0]
    balanced_majority = pd.DataFrame(X_resampled[majority_mask], columns=X_temp.columns)
    balanced_majority['class'] = majority_class_label  # Restore original type

    # print(f"Clase mayoritaria: {len(X_majority)} -> {len(balanced_majority)} muestras")
    # print(f"[Mayoritaria] Tamaño después de submuestreo: {len(balanced_majority)}")
    # print(f"[Mayoritaria] Reducción: {len(balanced_majority)/len(X_majority)*100:.1f}%")

    return balanced_majority


def minority_class_clustering(X_minority, max_clusters=6, min_cluster_obs=5, random_state=42):
    
    # print(f"Buscando número óptimo de clusters para la clase minoritaria...")

    # Search for optimal number of clusters
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, len(X_minority)))

    for n_clusters in cluster_range:
        if n_clusters >= len(X_minority):
            break

        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=1000,
            tol=0.0001,
            random_state=random_state,
            algorithm="lloyd"
        )
        labels = kmeans.fit_predict(X_minority)

        # Check if all clusters have minimum observations
        cluster_counts = np.bincount(labels)
        if np.any(cluster_counts < min_cluster_obs):
            silhouette_scores.append(-1)  # Penalize small clusters
            continue

        score = silhouette_score(X_minority, labels, metric='euclidean', random_state=random_state)
        silhouette_scores.append(score)

    # Determine optimal clusters
    if not silhouette_scores or all(score == -1 for score in silhouette_scores):
        optimal_clusters = 1
        kmeans_model = None
        # print(f"No se encontraron clusters válidos. Usando un solo cluster.")
    else:
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        kmeans_model = KMeans(
            n_clusters=optimal_clusters,
            init='k-means++',
            n_init=10,
            max_iter=1000,
            tol=0.0001,
            random_state=random_state,
            algorithm="lloyd"
        )
        kmeans_model.fit(X_minority)
        
        # Imprimir K optimo de silueta
        # print(f"[Minoritaria] K óptimo (Silhouete) = {optimal_clusters}")
        
        # print(f"[Minoritaria] Número óptimo de clusters: {optimal_clusters}")
        # print(f"Mejor índice silhouette: {max(silhouette_scores):.4f}")

    return optimal_clusters, kmeans_model


def minority_class_smote_balancing(X_minority, y_minority, X_majority_balanced,
                                 target_size, optimal_clusters, kmeans_model=None,
                                 min_cluster_obs=5, k_smote=3, random_state=42):
    
    # print(f"Aplicando balanceo SMOTE a la clase minoritaria...")

    minority_data = X_minority.copy()
    minority_class_label = y_minority.iloc[0]  # Get the actual label

    # Apply clustering
    if optimal_clusters == 1 or kmeans_model is None:
        minority_data['cluster'] = 0
        cluster_labels = [0]
    else:
        minority_data['cluster'] = kmeans_model.predict(X_minority)
        cluster_labels = list(range(optimal_clusters))

        # Merge small clusters with closest large clusters
        cluster_counts = minority_data['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < min_cluster_obs].index

        for small_cluster in small_clusters:
            if len(cluster_counts) > 1:
                large_clusters = cluster_counts[cluster_counts >= min_cluster_obs].index
                if len(large_clusters) > 0 and kmeans_model is not None:
                    small_centroid = kmeans_model.cluster_centers_[small_cluster]
                    distances = [np.linalg.norm(small_centroid - kmeans_model.cluster_centers_[c])
                               for c in large_clusters]
                    closest_cluster = large_clusters[np.argmin(distances)]
                    minority_data.loc[minority_data['cluster'] == small_cluster, 'cluster'] = closest_cluster
                    cluster_labels = [c for c in cluster_labels if c != small_cluster]

    # Get final cluster distribution
    cluster_distribution = minority_data['cluster'].value_counts()
    cluster_weights = cluster_distribution / cluster_distribution.sum()
    # Imprimir k optimo de silueta
    # print(f"[Minoritaria] K´ óptimo (Silhouete) = {len(cluster_labels)}")
    # print(f"Distribución de clusters: {dict(cluster_distribution)}")
    # print(f"Pesos de los clusters: {dict(cluster_weights)}")

    # Apply SMOTE per cluster
    balanced_minority_data = pd.DataFrame()

    for cluster_id in cluster_distribution.index:
        cluster_data = minority_data[minority_data['cluster'] == cluster_id].drop('cluster', axis=1)
        cluster_size = len(cluster_data)
        cluster_target_size = max(1, int(cluster_weights[cluster_id] * target_size))

        # print(f"Cluster {cluster_id}: {cluster_size} -> {cluster_target_size} muestras")

        if cluster_target_size <= cluster_size:
            # Subsample if target is smaller
            sampled_data = cluster_data.sample(n=cluster_target_size, random_state=random_state)
            balanced_minority_data = pd.concat([balanced_minority_data, sampled_data])
        else:
            # Apply SMOTE if we need more samples
            samples_needed = cluster_target_size - cluster_size

            if cluster_size < k_smote + 1:
                # Not enough samples for SMOTE, duplicate randomly
                # print(f"  Cluster demasiado pequeño para SMOTE, duplicando aleatoriamente")
                duplicated = cluster_data.sample(n=samples_needed, replace=True, random_state=random_state)
                cluster_result = pd.concat([cluster_data, duplicated])
            else:
                # Use SMOTE
                majority_fake = X_majority_balanced.sample(n=samples_needed * 2, replace=True, random_state=random_state)

                # Prepare data for SMOTE - Convert labels to strings to avoid type conflicts
                minority_class_str = str(minority_class_label)
                majority_class_str = 'majority_fake'

                X_smote = pd.concat([cluster_data, majority_fake])
                y_smote = [minority_class_str] * cluster_size + [majority_class_str] * len(majority_fake)

                # Apply SMOTE
                smote = SMOTE(random_state=random_state, k_neighbors=min(k_smote, cluster_size-1))
                X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)

                # Extract synthetic minority samples
                minority_mask = np.array(y_resampled) == minority_class_str
                resampled_minority = pd.DataFrame(X_resampled[minority_mask], columns=X_smote.columns)

                # Select target number of samples
                cluster_result = resampled_minority.sample(n=cluster_target_size, random_state=random_state)

            balanced_minority_data = pd.concat([balanced_minority_data, cluster_result])

    # Add class label (restore original type)
    balanced_minority_data['class'] = minority_class_label
    # print(f"[Minoritaria] Nuevas muestras sintéticas generadas: {len(balanced_minority_data) - len(X_minority)}")
    # print(f"[Minoritaria] Tamaño final (original + sintético): {len(balanced_minority_data)}")
    #print(f"Clase minoritaria: {len(X_minority)} -> {len(balanced_minority_data)} muestras")
    #print(f"Cambio: {len(balanced_minority_data)/len(X_minority)*100:.1f}%")

    return balanced_minority_data


def hcbou_balance(X, y, max_clusters_maj=8, max_clusters_min=6, k_smote=3,
                  min_cluster_obs=5, random_state=42, verbose=True):
    """
    Apply HCBOU (Hybrid Cluster-Based Oversampling and Undersampling) to balance dataset.

    This is the main function that implements the complete HCBOU pipeline:
    1. Identify majority and minority classes
    2. Apply cluster-based undersampling to majority class
    3. Apply cluster-based SMOTE oversampling to minority class
    4. Combine balanced classes

    Parameters:
    -----------
    X : pd.DataFrame
        Features matrix
    y : pd.Series
        Target labels
    max_clusters_maj : int, default=8
        Maximum clusters for majority class undersampling
    max_clusters_min : int, default=6
        Maximum clusters for minority class clustering
    k_smote : int, default=3
        Number of neighbors for SMOTE
    min_cluster_obs : int, default=5
        Minimum observations per cluster
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress information

    Returns:
    --------
    tuple
        (X_balanced, y_balanced) - Balanced features and labels

    Example:
    --------
    >>> from utils.hcbou import hcbou_balance
    >>> X_balanced, y_balanced = hcbou_balance(X_train, y_train, max_clusters_maj=10)
    """
    if verbose:
        
        print("[2] Implementación de HCBOU Iniciada")

    # Validate inputs
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    # Get class distribution
    class_counts = y.value_counts().sort_values(ascending=False)

    if len(class_counts) < 2:
        raise ValueError("Dataset must have at least 2 classes")

    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]

    if verbose:
        # print(f"Distribución original:")
        print(f"    Clase mayoritaria ({majority_class}): {class_counts[majority_class]} muestras")
        print(f"    Clase minoritaria ({minority_class}): {class_counts[minority_class]} muestras")
        print(f"    Total de clases: {class_counts.sum()}")
        print(f"    Ratio de desbalance: {class_counts[majority_class]/class_counts[minority_class]:.2f}")

    # Separate classes
    majority_mask = y == majority_class
    minority_mask = y == minority_class

    X_majority = X.loc[majority_mask]
    y_majority = y.loc[majority_mask]
    X_minority = X.loc[minority_mask]
    y_minority = y.loc[minority_mask]

    # Calculate target sizes for balancing
    total_samples = len(X)
    target_per_class = total_samples // 2  # For binary classification

    # Step 1: Majority class undersampling
    # if verbose:
    #    print(f"\nClase Mayoritarias---------")

    balanced_majority = majority_class_undersampling(
        X_majority, y_majority, target_per_class, max_clusters_maj, random_state
    )

    # Step 2: Minority class clustering and SMOTE balancing
    # if verbose:
    #    print(f"\nClase minoritarias---------")

    # Find optimal clusters
    optimal_clusters, kmeans_model = minority_class_clustering(
        X_minority, max_clusters_min, min_cluster_obs, random_state
    )

    # Apply SMOTE balancing
    balanced_minority = minority_class_smote_balancing(
        X_minority, y_minority, balanced_majority.drop('class', axis=1),
        target_per_class, optimal_clusters, kmeans_model,
        min_cluster_obs, k_smote, random_state
    )

    # Step 3: Combine balanced classes
    # if verbose:
        #print(f"\n🔄 Paso 3: Combinando clases balanceadas")
        #print("-" * 35)

    balanced_data = pd.concat([balanced_majority, balanced_minority], ignore_index=True)

    # Separate final features and labels
    X_balanced = balanced_data.drop('class', axis=1)
    y_balanced = balanced_data['class']

    # Final statistics
    final_counts = y_balanced.value_counts().sort_index()

    if verbose:
        # print(f"\nHCBOU balanceo terminado.")
        print(f"\nTamaño final del train balanceado: {len(y_balanced)}")
        #for class_label, count in final_counts.items():
        #    print(f"    Clase {class_label}: {count} muestras")
        print(f"Distribución clases: { {int(k): int(v) for k, v in final_counts.items()} }")
        # print(f"Ratio de balance: {final_counts.min()}:{final_counts.max()}")
        # print(f"Cambio de tamaño del conjunto de datos: {len(X)} -> {len(X_balanced)} ({(len(X_balanced)-len(X))/len(X)*100:+.1f}%)")

    return X_balanced, y_balanced


# Default configuration for different scenarios
DEFAULT_PARAMS = {
    'binary_classification': {
        'max_clusters_maj': 8,
        'max_clusters_min': 6,
        'k_smote': 3,
        'min_cluster_obs': 5,
    },
    'multiclass_small': {
        'max_clusters_maj': 6,
        'max_clusters_min': 4,
        'k_smote': 2,
        'min_cluster_obs': 3,
    },
    'multiclass_large': {
        'max_clusters_maj': 10,
        'max_clusters_min': 8,
        'k_smote': 5,
        'min_cluster_obs': 5,
    }
}


def get_recommended_params(X, y, scenario='auto'):
    
    n_classes = len(y.unique())
    n_samples = len(y)

    if scenario == 'auto':
        if n_classes == 2:
            scenario = 'binary_classification'
        elif n_samples < 1000:
            scenario = 'multiclass_small'
        else:
            scenario = 'multiclass_large'

    return DEFAULT_PARAMS.get(scenario, DEFAULT_PARAMS['binary_classification'])