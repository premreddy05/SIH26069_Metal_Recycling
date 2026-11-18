
#!/usr/bin/env python3
"""
SIH26069 - K-Means Clustering of Metal Recycling Material Streams

Project workflow:
1. Generate synthetic metal recycling batch data.
2. Clean and prepare the data.
3. Engineer features.
4. Find the best number of clusters.
5. Apply K-Means clustering.
6. Evaluate clustering performance.
7. Validate clusters using lab recovery values.
8. Generate and save six graphs.
9. Save the trained model and output files.
10. Automatically open the generated graphs.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Save graphs without requiring an interactive display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------
# 1. GENERATE SYNTHETIC DATA
# ---------------------------------------------------------

def generate_data(n_samples=1500, random_state=RANDOM_STATE):

    rng = np.random.default_rng(random_state)

    origins = [
        "Industrial Offcuts",
        "Construction/Demolition",
        "Household/Municipal",
        "Electronic Waste",
        "Automotive Scrap"
    ]

    origin_probabilities = [0.20, 0.25, 0.25, 0.15, 0.15]

    data = []

    for i in range(n_samples):

        origin = rng.choice(origins, p=origin_probabilities)

        if origin == "Industrial Offcuts":
            valuable = rng.uniform(65, 95)
            iron = rng.uniform(2, 15)
            hazard = rng.uniform(0, 5)
            contamination = rng.uniform(1, 8)

        elif origin == "Construction/Demolition":
            valuable = rng.uniform(5, 25)
            iron = rng.uniform(60, 90)
            hazard = rng.uniform(1, 8)
            contamination = rng.uniform(2, 12)

        elif origin == "Household/Municipal":
            valuable = rng.uniform(25, 65)
            iron = rng.uniform(15, 55)
            hazard = rng.uniform(5, 25)
            contamination = rng.uniform(8, 25)

        elif origin == "Electronic Waste":
            valuable = rng.uniform(45, 85)
            iron = rng.uniform(5, 30)
            hazard = rng.uniform(10, 30)
            contamination = rng.uniform(5, 20)

        else:
            valuable = rng.uniform(20, 55)
            iron = rng.uniform(25, 65)
            hazard = rng.uniform(2, 12)
            contamination = rng.uniform(5, 18)

        # Composition values
        copper = valuable * rng.uniform(0.20, 0.40)
        aluminium = valuable * rng.uniform(0.20, 0.40)
        other_valuable = max(valuable - copper - aluminium, 0)

        # Remaining composition
        remaining = max(100 - valuable, 0)

        iron = min(iron, remaining)

        plastic = remaining * rng.uniform(0.10, 0.30)
        other_material = max(remaining - iron - plastic, 0)

        # Composition percentages
        composition = np.array([
            copper,
            aluminium,
            other_valuable,
            iron,
            plastic,
            other_material
        ], dtype=float)

        # Normalize composition to 100%
        composition = composition / composition.sum() * 100

        copper, aluminium, other_valuable, iron, plastic, other_material = composition

        purity = valuable + rng.normal(0, 3)
        purity = np.clip(purity, 0, 100)

        moisture = rng.uniform(0, 20)
        density = rng.uniform(1, 8)
        batch_weight = rng.uniform(50, 1000)

        # Simulated laboratory recovery value
        lab_recovery = (
            0.55 * purity
            + 0.20 * valuable
            - 0.35 * hazard
            - 0.25 * contamination
            - 0.15 * moisture
            + rng.normal(0, 4)
        )

        lab_recovery = np.clip(lab_recovery, 0, 100)

        data.append({
            "batch_id": f"B{i + 1:04d}",
            "source_origin": origin,
            "copper_pct": copper,
            "aluminium_pct": aluminium,
            "other_valuable_pct": other_valuable,
            "iron_pct": iron,
            "plastic_pct": plastic,
            "other_material_pct": other_material,
            "purity_pct": purity,
            "valuable_metal_pct": valuable,
            "hazard_score": hazard,
            "contamination_pct": contamination,
            "moisture_pct": moisture,
            "density": density,
            "batch_weight": batch_weight,
            "lab_recovery_value": lab_recovery
        })

    return pd.DataFrame(data)


# ---------------------------------------------------------
# 2. VALIDATE AND CLEAN DATA
# ---------------------------------------------------------

def clean_data(df):

    required_columns = [
        "batch_id",
        "source_origin",
        "copper_pct",
        "aluminium_pct",
        "other_valuable_pct",
        "iron_pct",
        "plastic_pct",
        "other_material_pct",
        "purity_pct",
        "valuable_metal_pct",
        "hazard_score",
        "contamination_pct",
        "moisture_pct",
        "density",
        "batch_weight",
        "lab_recovery_value"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df = df.drop_duplicates(subset=["batch_id"])

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:
        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        df[col] = df[col].fillna(df[col].median())

    df["source_origin"] = (
        df["source_origin"]
        .fillna("Unknown")
        .astype(str)
    )

    return df


# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------

def engineer_features(df):

    df = df.copy()

    df["valuable_to_iron_ratio"] = (
        df["valuable_metal_pct"]
        / (df["iron_pct"] + 0.01)
    )

    df["contamination_to_purity_ratio"] = (
        df["contamination_pct"]
        / (df["purity_pct"] + 0.01)
    )

    df["recyclability_score"] = (
        0.40 * df["purity_pct"]
        + 0.30 * df["valuable_metal_pct"]
        - 0.15 * df["hazard_score"]
        - 0.15 * df["contamination_pct"]
    )

    df["total_composition"] = (
        df["copper_pct"]
        + df["aluminium_pct"]
        + df["other_valuable_pct"]
        + df["iron_pct"]
        + df["plastic_pct"]
        + df["other_material_pct"]
    )

    return df


# ---------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------

def create_preprocessor(df):

    numeric_features = [
        "copper_pct",
        "aluminium_pct",
        "other_valuable_pct",
        "iron_pct",
        "plastic_pct",
        "other_material_pct",
        "purity_pct",
        "valuable_metal_pct",
        "hazard_score",
        "contamination_pct",
        "moisture_pct",
        "density",
        "batch_weight",
        "valuable_to_iron_ratio",
        "contamination_to_purity_ratio",
        "recyclability_score"
    ]

    categorical_features = ["source_origin"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    X = preprocessor.fit_transform(df)

    feature_count = X.shape[1]

    print(
        f"[features] {feature_count} model features: "
        f"{len(numeric_features)} numeric + origin one-hot"
    )

    return X, preprocessor, numeric_features


# ---------------------------------------------------------
# 5. FIND OPTIMAL NUMBER OF CLUSTERS
# ---------------------------------------------------------

def evaluate_k_range(X, k_min=2, k_max=10):

    results = []

    for k in range(k_min, k_max + 1):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)

        ch_score = calinski_harabasz_score(X, labels)

        db_score = davies_bouldin_score(X, labels)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette,
            "calinski_harabasz": ch_score,
            "davies_bouldin": db_score
        })

    return pd.DataFrame(results)


def choose_best_k(results):

    results = results.copy()

    # Normalize metrics into rankings.
    results["silhouette_rank"] = (
        results["silhouette"].rank(ascending=False)
    )

    results["ch_rank"] = (
        results["calinski_harabasz"].rank(ascending=False)
    )

    results["db_rank"] = (
        results["davies_bouldin"].rank(ascending=True)
    )

    results["average_rank"] = (
        results["silhouette_rank"]
        + results["ch_rank"]
        + results["db_rank"]
    ) / 3

    best_row = results.loc[
        results["average_rank"].idxmin()
    ]

    best_k = int(best_row["k"])

    print(f"[kmeans] chosen k = {best_k}")

    return best_k


# ---------------------------------------------------------
# 6. K-MEANS CLUSTERING
# ---------------------------------------------------------

def apply_kmeans(X, k):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=500
    )

    labels = model.fit_predict(X)

    print("[kmeans] clustering completed")

    print(
        "[kmeans] inertia = "
        f"{model.inertia_:.3f}"
    )

    print(
        "[kmeans] silhouette = "
        f"{silhouette_score(X, labels):.3f}"
    )

    print(
        "[kmeans] Calinski-Harabasz = "
        f"{calinski_harabasz_score(X, labels):.3f}"
    )

    print(
        "[kmeans] Davies-Bouldin = "
        f"{davies_bouldin_score(X, labels):.3f}"
    )

    return model, labels


# ---------------------------------------------------------
# 7. CLUSTER STABILITY
# ---------------------------------------------------------

def calculate_stability(X, labels, k, repetitions=10):

    rng = np.random.default_rng(RANDOM_STATE)

    scores = []

    n_samples = X.shape[0]

    for _ in range(repetitions):

        sample_size = int(n_samples * 0.80)

        indices = rng.choice(
            n_samples,
            size=sample_size,
            replace=False
        )

        X_sample = X[indices]

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        sample_labels = model.fit_predict(X_sample)

        original_labels = labels[indices]

        score = adjusted_rand_score(
            original_labels,
            sample_labels
        )

        scores.append(score)

    mean_score = float(np.mean(scores))

    print(
        "[kmeans] bootstrap stability "
        f"(mean ARI) = {mean_score:.3f}"
    )

    return mean_score


# ---------------------------------------------------------
# 8. ASSIGN RECYCLABILITY TIERS
# ---------------------------------------------------------

def assign_tiers(df):

    df = df.copy()

    profiles = df.groupby("cluster").agg({
        "purity_pct": "mean",
        "valuable_metal_pct": "mean",
        "hazard_score": "mean",
        "contamination_pct": "mean",
        "lab_recovery_value": "mean"
    })

    profiles["tier_score"] = (
        profiles["purity_pct"] * 0.35
        + profiles["valuable_metal_pct"] * 0.35
        - profiles["hazard_score"] * 0.15
        - profiles["contamination_pct"] * 0.15
    )

    ordered_clusters = (
        profiles["tier_score"]
        .sort_values(ascending=False)
        .index.tolist()
    )

    tier_names = [
        "Tier 1 - High Recyclability",
        "Tier 2 - Medium Recyclability",
        "Tier 3 - Low Recyclability"
    ]

    tier_mapping = {}

    for index, cluster in enumerate(ordered_clusters):

        if index < len(tier_names):
            tier_mapping[cluster] = tier_names[index]
        else:
            tier_mapping[cluster] = (
                f"Tier {index + 1} - Recyclability"
            )

    df["recyclability_tier"] = (
        df["cluster"].map(tier_mapping)
    )

    return df, profiles, tier_mapping


# ---------------------------------------------------------
# 9. LAB VALIDATION
# ---------------------------------------------------------

def validate_against_lab(df):

    df = df.copy()

    df["lab_band"] = pd.cut(
        df["lab_recovery_value"],
        bins=[-np.inf, 40, 65, np.inf],
        labels=[
            "Low Lab Recovery",
            "Medium Lab Recovery",
            "High Lab Recovery"
        ]
    )

    validation = df.groupby(
        "recyclability_tier",
        observed=False
    ).agg(
        sample_count=("batch_id", "count"),
        mean_lab_recovery=("lab_recovery_value", "mean"),
        mean_purity=("purity_pct", "mean"),
        mean_valuable_metal=(
            "valuable_metal_pct",
            "mean"
        )
    ).reset_index()

    print("\n[validation] Laboratory recovery validation")

    print(validation.to_string(index=False))

    return df, validation


# ---------------------------------------------------------
# 10. CLUSTER PROFILE REPORT
# ---------------------------------------------------------

def print_cluster_profiles(df):

    print("\n[profiles] Recyclability tier profiles")

    profile = df.groupby(
        "recyclability_tier",
        observed=False
    ).agg(
        sample_count=("batch_id", "count"),
        purity=("purity_pct", "mean"),
        valuable_metal=("valuable_metal_pct", "mean"),
        iron=("iron_pct", "mean"),
        hazard=("hazard_score", "mean"),
        contamination=("contamination_pct", "mean"),
        lab_mean=("lab_recovery_value", "mean")
    )

    for tier, row in profile.iterrows():

        dominant_origin = (
            df[df["recyclability_tier"] == tier]
            ["source_origin"]
            .mode()
        )

        if len(dominant_origin) > 0:
            origin = dominant_origin.iloc[0]
        else:
            origin = "Unknown"

        print(f"\n{tier}")
        print(f"  Samples: {int(row['sample_count'])}")
        print(f"  Purity: {row['purity']:.2f}")
        print(
            f"  Valuable metal: "
            f"{row['valuable_metal']:.2f}"
        )
        print(f"  Iron: {row['iron']:.2f}")
        print(f"  Hazard: {row['hazard']:.2f}")
        print(
            f"  Contamination: "
            f"{row['contamination']:.2f}"
        )
        print(f"  Lab recovery: {row['lab_mean']:.2f}")
        print(f"  Dominant origin: {origin}")

    return profile


# ---------------------------------------------------------
# 11. GRAPH 1 - K SELECTION
# ---------------------------------------------------------

def plot_k_selection(results):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9)
    )

    axes[0, 0].plot(
        results["k"],
        results["inertia"],
        marker="o"
    )

    axes[0, 0].set_title("Elbow Method")
    axes[0, 0].set_xlabel("Number of Clusters")
    axes[0, 0].set_ylabel("Inertia")

    axes[0, 1].plot(
        results["k"],
        results["silhouette"],
        marker="o"
    )

    axes[0, 1].set_title("Silhouette Score")
    axes[0, 1].set_xlabel("Number of Clusters")
    axes[0, 1].set_ylabel("Score")

    axes[1, 0].plot(
        results["k"],
        results["calinski_harabasz"],
        marker="o"
    )

    axes[1, 0].set_title("Calinski-Harabasz Score")
    axes[1, 0].set_xlabel("Number of Clusters")
    axes[1, 0].set_ylabel("Score")

    axes[1, 1].plot(
        results["k"],
        results["davies_bouldin"],
        marker="o"
    )

    axes[1, 1].set_title("Davies-Bouldin Score")
    axes[1, 1].set_xlabel("Number of Clusters")
    axes[1, 1].set_ylabel("Score")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "plot_k_selection.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 12. GRAPH 2 - PCA CLUSTER VISUALIZATION
# ---------------------------------------------------------

def plot_pca_clusters(X, labels):

    pca = PCA(n_components=2)

    components = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)

    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "cluster": labels.astype(str)
    })

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="viridis",
        s=35,
        alpha=0.75
    )

    plt.title("K-Means Clusters using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "plot_pca_tiers.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 13. GRAPH 3 - SILHOUETTE SCORE
# ---------------------------------------------------------

def plot_silhouette(results):

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=results,
        x="k",
        y="silhouette"
    )

    plt.title("Silhouette Score for Different K Values")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "plot_silhouette.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 14. GRAPH 4 - LAB VALIDATION
# ---------------------------------------------------------

def plot_lab_validation(df):

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="recyclability_tier",
        y="lab_recovery_value"
    )

    plt.title(
        "Laboratory Recovery Value by Recyclability Tier"
    )

    plt.xlabel("Recyclability Tier")
    plt.ylabel("Lab Recovery Value")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "plot_lab_validation.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 15. GRAPH 5 - TIER PROFILES
# ---------------------------------------------------------

def plot_tier_profiles(df):

    columns = [
        "purity_pct",
        "valuable_metal_pct",
        "iron_pct",
        "hazard_score",
        "contamination_pct",
        "lab_recovery_value"
    ]

    profile = df.groupby(
        "recyclability_tier",
        observed=False
    )[columns].mean()

    plt.figure(figsize=(12, 7))

    sns.heatmap(
        profile,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5
    )

    plt.title("Average Feature Profile of Recyclability Tiers")
    plt.xlabel("Material Features")
    plt.ylabel("Recyclability Tier")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "plot_tier_profiles.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 16. GRAPH 6 - SOURCE ORIGIN DISTRIBUTION
# ---------------------------------------------------------

def plot_origin_mix(df):

    origin_data = pd.crosstab(
        df["recyclability_tier"],
        df["source_origin"]
    )

    plt.figure(figsize=(12, 7))

    origin_data.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 7)
    )

    plt.title(
        "Source Origin Distribution Across Recyclability Tiers"
    )

    plt.xlabel("Recyclability Tier")
    plt.ylabel("Number of Batches")

    plt.xticks(rotation=15)

    plt.legend(
        title="Source Origin",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "plot_origin_mix.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 17. SAVE PROJECT OUTPUTS
# ---------------------------------------------------------

def save_outputs(df, results, validation, model, preprocessor,
                 stability, tier_mapping):

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df.to_csv(
        OUTPUT_DIR / "clustered_metal_batches.csv",
        index=False
    )

    results.to_csv(
        OUTPUT_DIR / "k_evaluation_results.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "lab_validation_results.csv",
        index=False
    )

    joblib.dump(
        model,
        MODEL_DIR / "kmeans_model.pkl"
    )

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl"
    )

    with open(
        OUTPUT_DIR / "project_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump({
            "selected_k": int(model.n_clusters),
            "cluster_stability_ari": float(stability),
            "tier_mapping": {
                str(k): str(v)
                for k, v in tier_mapping.items()
            },
            "total_batches": int(len(df))
        }, file, indent=4)

    print("\n[save] CSV files saved")
    print("[save] Model saved")
    print("[save] Preprocessor saved")
    print("[save] Project summary saved")


# ---------------------------------------------------------
# 18. OPEN GENERATED GRAPHS
# ---------------------------------------------------------

def open_generated_graphs():

    graph_names = [
        "plot_k_selection.png",
        "plot_pca_tiers.png",
        "plot_silhouette.png",
        "plot_lab_validation.png",
        "plot_tier_profiles.png",
        "plot_origin_mix.png"
    ]

    print("\n[graphs] Opening generated graphs...")

    for graph_name in graph_names:

        graph_path = OUTPUT_DIR / graph_name

        if graph_path.exists():

            try:
                os.startfile(str(graph_path))
                print(f"[opened] {graph_name}")

            except Exception as error:
                print(
                    f"[warning] Could not open "
                    f"{graph_name}: {error}"
                )


# ---------------------------------------------------------
# 19. MAIN FUNCTION
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("SIH26069 - METAL RECYCLING K-MEANS CLUSTERING")
    print("=" * 65)

    # Step 1: Generate data
    df = generate_data(n_samples=1500)

    print(
        f"\n[data] Generated {len(df):,} synthetic batches"
    )

    # Step 2: Clean data
    df = clean_data(df)

    print("[data] Data cleaning completed")

    # Step 3: Feature engineering
    df = engineer_features(df)

    print("[features] Feature engineering completed")

    # Step 4: Preprocessing
    X, preprocessor, numeric_features = (
        create_preprocessor(df)
    )

    # Step 5: Evaluate different K values
    print("\n[kmeans] Evaluating cluster numbers...")

    results = evaluate_k_range(
        X,
        k_min=2,
        k_max=10
    )

    # Step 6: Choose best K
    best_k = choose_best_k(results)

    # Step 7: Apply K-Means
    model, labels = apply_kmeans(X, best_k)

    df["cluster"] = labels

    # Step 8: Stability evaluation
    stability = calculate_stability(
        X,
        labels,
        best_k,
        repetitions=10
    )

    # Step 9: Assign tiers
    df, cluster_profiles, tier_mapping = (
        assign_tiers(df)
    )

    # Step 10: Lab validation
    df, validation = validate_against_lab(df)

    # Step 11: Print cluster profiles
    profile = print_cluster_profiles(df)

    # Step 12: Generate graphs
    print("\n[plots] Generating graphs...")

    plot_k_selection(results)

    plot_pca_clusters(X, labels)

    plot_silhouette(results)

    plot_lab_validation(df)

    plot_tier_profiles(df)

    plot_origin_mix(df)

    # Step 13: Save outputs
    save_outputs(
        df,
        results,
        validation,
        model,
        preprocessor,
        stability,
        tier_mapping
    )

    # Step 14: Open generated graphs
    open_generated_graphs()

    print("\n" + "=" * 65)
    print("[done] All artifacts saved in ./outputs/")
    print("[done] Metal recycling clustering completed")
    print("=" * 65)


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main()




