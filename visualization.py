import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import ConvexHull
from preprocessing import VANCOUVER_LAT_BOUNDS, VANCOUVER_LON_BOUNDS, data
from scipy.stats import zscore

COLORS = [
    "#D62728",
    "#1F77B4",
    "#2CA02C",
    "#F37100", 
    "#8931D1",
    "#81542C",
    "#A1991C",
    "#17BECF",
    "#F800EC",
    "#520000",
    "#276814",
    "#2D3BB6",
]


def plot_neighbourhoods(max_z_score= 1.25):
    df = data.copy()
    df["lat_zscore"] = df.groupby("NEIGHBOURHOOD_CODE")["latitude"].transform(zscore)
    df["lon_zscore"] = df.groupby("NEIGHBOURHOOD_CODE")["longitude"].transform(zscore)

    df_filtered = df[(df["lat_zscore"].abs() <= max_z_score) & (df["lon_zscore"].abs() <= max_z_score)].copy()
    df_filtered.drop(columns=["lat_zscore", "lon_zscore"], inplace=True)

    grouped_filtered = df_filtered.groupby("NEIGHBOURHOOD_CODE").agg(
        lat_min=("latitude", "min"),
        lat_max=("latitude", "max"),
        lat_mean=("latitude", "mean"),
        lon_min=("longitude", "min"),
        lon_max=("longitude", "max"),
        lon_mean=("longitude", "mean"),
        postal_codes=("PROPERTY_POSTAL_CODE", "count"),
        property_count=("PID", "count"),
    ).reset_index()

    return plot_points_and_hulls(df_filtered, grouped_filtered)

def plot_points_and_hulls(df_points, summary):
    data = df_points[["NEIGHBOURHOOD_CODE", "latitude", "longitude"]].dropna()

    codes = sorted(data["NEIGHBOURHOOD_CODE"].unique())
    color_map = {code: COLORS[i % len(COLORS)] for i, code in enumerate(codes)}

    fig, ax = plt.subplots(figsize=(20, 10))
    img = mpimg.imread("map_bounded.png")
    ax.imshow(
        img,
        extent=[*VANCOUVER_LON_BOUNDS, *VANCOUVER_LAT_BOUNDS],
        aspect="auto",
        zorder=1,
    )

    for _, row in summary.iterrows():
        ax.text(
            row["lon_mean"],
            row["lat_mean"],
            int(row["NEIGHBOURHOOD_CODE"]),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6),
            zorder=4,
        )

    for code, group in data.groupby("NEIGHBOURHOOD_CODE"):
        pts = group[["longitude", "latitude"]].to_numpy()
        if len(pts) < 3:
            continue
        hull = ConvexHull(pts, qhull_options="QJ")
        hull_points = pts[hull.vertices]
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)

        ax.fill(
            hull_points[:, 0],
            hull_points[:, 1],
            color=color_map[code],
            alpha=0.2,
            zorder=2,
        )
        ax.plot(
            hull_points[:, 0],
            hull_points[:, 1],
            color=color_map[code],
            linewidth=1.5,
            alpha=0.9,
            zorder=5,
        )

    ax.set_xlim(*VANCOUVER_LON_BOUNDS)
    ax.set_ylim(*VANCOUVER_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("BCAssessment Neighbourhood Code Visualization")
    ax.grid(alpha=0.3)
    return fig