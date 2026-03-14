from __future__ import annotations

import matplotlib.pyplot as plt


def plot_heatmap(data, norm_data, force: int = 15, sensor_id: int = 1):
    data_filtered = data[data["F"] == force][["X", "Y", f"Sensor R{sensor_id}"]].reset_index(drop=True)
    heatmap_data = data_filtered.pivot(index="Y", columns="X", values=f"Sensor R{sensor_id}")
    norm_data_filtered = norm_data[norm_data["F"] == force][["X", "Y", f"Sensor R{sensor_id}"]].reset_index(drop=True)
    norm_heatmap_data = norm_data_filtered.pivot(index="Y", columns="X", values=f"Sensor R{sensor_id}")

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(
        heatmap_data,
        origin="lower",
        cmap="viridis",
        extent=[
            data_filtered["X"].min(),
            data_filtered["X"].max(),
            data_filtered["Y"].min(),
            data_filtered["Y"].max(),
        ],
    )
    plt.colorbar(im, ax=ax1, label=f"Sensor R{sensor_id}")
    ax1.set_title(f"Heatmap for Sensor R{sensor_id} at Force {force}N")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(
        norm_heatmap_data,
        origin="lower",
        cmap="viridis",
        extent=[
            norm_data_filtered["X"].min(),
            norm_data_filtered["X"].max(),
            norm_data_filtered["Y"].min(),
            norm_data_filtered["Y"].max(),
        ],
    )
    plt.colorbar(im2, ax=ax2, label=f"Sensor R{sensor_id}")
    ax2.set_title(f"Normalized Heatmap for Sensor R{sensor_id} at Force {force}N")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    plt.show()

