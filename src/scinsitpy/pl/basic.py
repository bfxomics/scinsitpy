import base64
import json
import zlib

import anndata as an
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from clustergrammer2 import CGM2, Network
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mplPolygon
from observable_jupyter import embed
from skimage import exposure, img_as_float

from scinsitpy.io.basic import get_palette, load_bounds_pixel


def view_region(
    adata: an.AnnData,
    library_id: str,
    color_key: str,
    geneNames: tuple,
    pt_size: float,
    x: int,
    y: int,
    size_x: int,
    size_y: int,
    metafilt: str = None,
    metafiltvals: str = None,
    all_transcripts: bool = False,
    noimage: bool = False,
    save: bool = False,
    figsize: tuple = (8, 8),
) -> an.AnnData:
    """Scinsit crop region plot.

    Parameters
    ----------
    adata
        Anndata object.
    library_id
        library id.

    Returns
    -------
    Return the AnnData object of the crop region.
    """
    img = sq.im.ImageContainer(adata.uns["spatial"][library_id]["images"]["hires"], library_id=library_id)

    if (img.shape[0] < y + size_y) or (img.shape[1] < x + size_x):
        print("Crop outside range img width/height = ", img.shape)
        return 0
    else:
        crop1 = img.crop_corner(y, x, size=(size_y, size_x))
        adata_crop = crop1.subset(adata, spatial_key="pixel")

        if adata_crop.shape[0] == 0:
            print("No cells found in region [x=", x, ", y=", y, ", size_x=", size_x, ", size_y=", size_y)
            return adata_crop
        else:
            if metafilt is not None:
                adata_crop = adata_crop[adata_crop.obs[metafilt].isin(metafiltvals)]

            print(adata_crop.shape[0], "cells to plot")

            # load cell boundaries
            adata_crop = load_bounds_pixel(adata_crop, library_id)

            currentCells = []
            typedCells = []
            for cell_id in adata_crop.obs.index:
                currentCells.append(adata_crop.obs.bounds[cell_id])
                typedCells.append(adata_crop.obs[color_key][cell_id])

            # minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
            # maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)

            # segmentation data
            polygon_data = []
            for inst_index in range(len(currentCells)):
                inst_cell = currentCells[inst_index]
                df_poly_z = pd.DataFrame(inst_cell).transpose()
                df_poly_z.columns.tolist()

                xxx = df_poly_z.values - [x, y]
                inst_poly = np.array(xxx.tolist())
                polygon_data.append(inst_poly)

            # generate colors for categories by plotting
            cats = adata_crop.obs[color_key].cat.categories.tolist()
            colors = list(adata_crop.uns[color_key + "_colors"])
            cat_colors = dict(zip(cats, colors))
            ser_color = pd.Series(cat_colors)

            fig, ax = plt.subplots(figsize=figsize)

            if noimage is False:
                crop1.show(layer="image", ax=ax)

            # plot cells
            pts = adata_crop.obs[["x_pix", "y_pix", color_key]]
            pts.x_pix -= x
            pts.y_pix -= y
            ax = sns.scatterplot(
                x="x_pix", y="y_pix", s=0, alpha=0.5, edgecolors=color_key, data=pts, hue=color_key, palette=cat_colors
            )

            # plot polygons
            polygons = [mplPolygon(polygon_data[i].reshape(-1, 2)) for i in range(len(polygon_data))]
            ax.add_collection(PatchCollection(polygons, fc=ser_color[typedCells], ec="w", alpha=0.8, linewidths=0.2))

            # plot transcripts
            transcripts = adata_crop.uns["transcripts"][library_id]
            tr = transcripts[
                (transcripts.x_pix > x)
                & (transcripts.x_pix < x + size_x)
                & (transcripts.y_pix > y)
                & (transcripts.y_pix < y + size_y)
            ]
            tr.loc[:, "x_pix"] = tr.loc[:, "x_pix"] - x
            tr.loc[:, "y_pix"] = tr.loc[:, "y_pix"] - y
            pts = tr[["x_pix", "y_pix"]].to_numpy()

            print(tr.gene.value_counts().head(10))

            if all_transcripts:
                plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="grey", s=pt_size, label="all")

            i = 0
            cols = sns.color_palette("deep", 10)
            for gn in geneNames:
                tr2 = tr[tr.gene == gn]
                pts = tr2[["x_pix", "y_pix"]].to_numpy()
                pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < size_x) & (pts[:, 1] < size_y)]
                plt.scatter(pts[:, 0], pts[:, 1], marker="x", color=cols[i], s=pt_size, label=gn)
                i = i + 1

            h, l = ax.get_legend_handles_labels()
            l1 = ax.legend(
                h[: len(cat_colors)], l[: len(cat_colors)], loc="upper right", bbox_to_anchor=(1, 1), fontsize=8
            )
            ax.legend(h[len(cat_colors) :], l[len(cat_colors) :], loc="upper left", bbox_to_anchor=(0, 1), fontsize=8)
            ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first
            title = (
                library_id + "[x=" + str(x) + ",y=" + str(y) + ",size_x=" + str(size_x) + ",size_y=" + str(size_y) + "]"
            )
            plt.title(title)
            plt.tight_layout()

            if save is True:
                print("saving " + title + ".pdf")
                plt.savefig(title + ".pdf", format="pdf", bbox_inches="tight")

    return adata_crop


def view_qc(adata: an.AnnData, library_id: str) -> int:
    """Scinsit quality control plot.

    Parameters
    ----------
    adata
        Anndata object.
    library_id
        library id.

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplot(2, 2, 1)
    bins = np.logspace(0, 4, 100)
    plt.hist(adata.obs["volume"], alpha=0.2, bins=bins, label=library_id, color="red")
    plt.xlabel("Volume")
    plt.ylabel("Cell count")
    plt.xscale("log")
    # Transcript count by cell
    plt.subplot(2, 2, 2)
    bins = np.logspace(0, 4, 100)
    plt.hist(adata.obs["barcodeCount"], alpha=0.2, bins=bins, label=library_id, color="red")
    plt.xlabel("Transcript count")
    plt.ylabel("Cell count")
    plt.xscale("log")
    plt.yscale("log")
    plt.subplot(2, 2, 3)
    barcodeCount = adata.obs["barcodeCount"]
    sns.distplot(barcodeCount, label=library_id, color="red")
    ax1 = plt.subplot(2, 2, 4)
    sc.pl.violin(adata, keys="barcodeCount", ax=ax1)

    return 0


def view_pop(
    adata: an.AnnData, celltypelabel: str = "celltype", metalabel: str = "population", figsize: tuple = (20, 5)
):
    """Scinsit compartment plot.

    Parameters
    ----------
    adata
        Anndata object.
    library_id
        library id.

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    ad = adata[adata.obs[metalabel].isin(["Stroma"])]
    sns.scatterplot(
        x="x_pix", y="y_pix", data=ad.obs, s=2, hue=celltypelabel, ax=ax1, palette=get_palette(celltypelabel)
    )
    ad = adata[adata.obs[metalabel].isin(["Epithelial"])]
    sns.scatterplot(
        x="x_pix", y="y_pix", data=ad.obs, s=2, hue=celltypelabel, ax=ax2, palette=get_palette(celltypelabel)
    )
    ad = adata[adata.obs[metalabel].isin(["Immune"])]
    sns.scatterplot(
        x="x_pix", y="y_pix", data=ad.obs, s=2, hue=celltypelabel, ax=ax3, palette=get_palette(celltypelabel)
    )
    ad = adata[adata.obs[metalabel].isin(["Endothelial"])]
    sns.scatterplot(
        x="x_pix", y="y_pix", data=ad.obs, s=2, hue=celltypelabel, ax=ax4, palette=get_palette(celltypelabel)
    )
    plt.setp(ax1.get_legend().get_texts(), fontsize="6")
    plt.setp(ax2.get_legend().get_texts(), fontsize="6")
    plt.setp(ax3.get_legend().get_texts(), fontsize="6")
    plt.setp(ax4.get_legend().get_texts(), fontsize="6")


def json_zip(j):
    """Json zipper"""
    zip_json_string = base64.b64encode(zlib.compress(json.dumps(j).encode("utf-8"))).decode("ascii")
    return zip_json_string


def embed_vizgen(adata: an.AnnData, color_key: str):
    """Embed vizgen assay (Nicolas Fernandez credit).

    Parameters
    ----------
    adata
        Anndata object.
    color_key
        color key.

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    cats = adata.obs[color_key].cat.categories.tolist()
    colors = list(adata.uns[color_key + "_colors"])
    cat_colors = dict(zip(cats, colors))
    ser_color = pd.Series(cat_colors)
    ser_color.name = "color"
    df_colors = pd.DataFrame(ser_color)
    df_colors.index = [str(x) for x in df_colors.index.tolist()]

    ser_counts = adata.obs[color_key].value_counts()
    ser_counts.name = "cell counts"
    meta_leiden = pd.DataFrame(ser_counts)
    sig_leiden = pd.DataFrame(columns=adata.var_names, index=adata.obs[color_key].cat.categories)
    for clust in adata.obs[color_key].cat.categories:
        sig_leiden.loc[clust] = adata[adata.obs[color_key].isin([clust]), :].X.mean(0)

    sig_leiden = sig_leiden.transpose()
    leiden_clusters = [str(x) for x in sig_leiden.columns.tolist()]
    sig_leiden.columns = leiden_clusters
    meta_leiden.index = sig_leiden.columns.tolist()
    meta_leiden[color_key] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())

    net = Network(CGM2)
    # net.load_df(sig_leiden, meta_col=meta_leiden, col_cats=['cell counts'])
    net.load_df(
        sig_leiden,
        meta_col=meta_leiden,
        col_cats=[color_key, "cell counts"],
        meta_row=adata.var,
        row_cats=["mean", "expression"],
    )
    net.filter_threshold(0.01, axis="row")
    net.normalize(axis="row", norm_type="zscore")
    net.set_global_cat_colors(df_colors)
    net.cluster()

    gex_int = pd.DataFrame(adata.layers["counts"], index=adata.obs.index.copy(), columns=adata.var_names).astype(np.int)
    gex_dict = {}
    for inst_gene in gex_int.columns.tolist():
        if "Blank" not in inst_gene:
            ser_gene = gex_int[inst_gene]
            ser_gene = ser_gene[ser_gene > 0]
            ser_gene = ser_gene.astype(np.int8)
            gex_dict[inst_gene] = ser_gene.to_dict()

    df_pos = adata.obs[["center_x", "center_y", color_key]]
    df_pos[["center_x", "center_y"]] = df_pos[["center_x", "center_y"]].round(2)
    df_pos.columns = ["x", "y", "leiden"]
    df_pos["y"] = -df_pos["y"]
    df_umap = adata.obsm.to_df()[["X_umap1", "X_umap2"]].round(2)
    df_umap.columns = ["umap-x", "umap-y"]

    # rotate the mouse brain to the upright position
    # theta = np.deg2rad(-15)
    # rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    # df_pos[['x', 'y']] = df_pos[['x', 'y']].dot(rot)

    df_name = pd.DataFrame(df_pos.index.tolist(), index=df_pos.index.tolist(), columns=["name"])

    df_obs = pd.concat([df_name, df_pos, df_umap], axis=1)
    data = df_obs.to_dict("records")

    obs_data = {"gex_dict": gex_dict, "data": data, "cat_colors": cat_colors, "network": net.viz}

    zip_obs_data = json_zip(obs_data)

    inputs = {
        "zoom": -3.5,
        "ini_cat": color_key,
        "ini_map_type": "UMAP",
        "ini_min_radius": 1.75,
        "zip_obs_data": zip_obs_data,
        "gex_opacity_contrast_scale": 0.85,
    }

    embed(
        "@vizgen/umap-spatial-heatmap-single-cell-0-3-0",
        cells=["viewof cgm", "dashboard"],
        inputs=inputs,
        display_logo=False,
    )


def getFovCoordinates(fov: int, meta_cell: pd.DataFrame) -> tuple:
    """Return fov coordinates

    Parameters
    ----------
    fov
        fov number.
    meta_cell
        cell metadata.

    Returns
    -------
    Return fov coordinates.
    """
    xmin = meta_cell.x[meta_cell.fov == fov].min()
    ymin = meta_cell.y[meta_cell.fov == fov].min()
    xmax = meta_cell.x[meta_cell.fov == fov].max()
    ymax = meta_cell.y[meta_cell.fov == fov].max()

    return (xmin, ymin, xmax, ymax)


def plot_img_and_hist(image: np.ndarray, axes: int, bins: int = 256):
    """Plot an image along with its histogram and cumulative histogram."""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype="step", color="black")
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot_contrast_panels(img: np.ndarray) -> int:
    """Test different contrasts for image received."""
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title("Contrast stretching")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title("Histogram equalization")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title("Adaptive equalization")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1


def plot_gamma_panels(img: np.ndarray) -> int:
    """Test different correction for image received."""
    # Gamma
    gamma_corrected = exposure.adjust_gamma(img, 2)

    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 1)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=object)
    axes[0, 0] = plt.subplot(2, 3, 1)
    axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = plt.subplot(2, 3, 4)
    axes[1, 1] = plt.subplot(2, 3, 5)
    axes[1, 2] = plt.subplot(2, 3, 6)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
    ax_img.set_title("Gamma correction")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])
    ax_img.set_title("Logarithmic correction")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1


def plot_adaptive_size_panels(img: np.ndarray, clip_limit: float) -> int:
    """Test different kernel_size for exposure.equalize_adapthist of image received."""
    # Adaptive Equalization
    img_1 = exposure.equalize_adapthist(img, clip_limit=clip_limit)
    img_2 = exposure.equalize_adapthist(img, clip_limit=clip_limit, kernel_size=[100, 100])
    img_3 = exposure.equalize_adapthist(img, clip_limit=clip_limit, kernel_size=[1000, 1000])

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_1, axes[:, 1])
    ax_img.set_title("adapt " + str(clip_limit) + ", default size 1/8")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_2, axes[:, 2])
    ax_img.set_title("adapt " + str(clip_limit) + ", size 100p")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_3, axes[:, 3])
    ax_img.set_title("adapt " + str(clip_limit) + ", size 1kp")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1


def plot_adaptive_panels(img: np.ndarray) -> int:
    """Test different clip_limit for exposure.equalize_adapthist of image received."""
    # Adaptive Equalization
    img_001 = exposure.equalize_adapthist(img, clip_limit=0.01)
    img_003 = exposure.equalize_adapthist(img, clip_limit=0.03)
    img_01 = exposure.equalize_adapthist(img, clip_limit=0.1)

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_001, axes[:, 1])
    ax_img.set_title("adapt 0.01")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_003, axes[:, 2])
    ax_img.set_title("adapt 0.03")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_01, axes[:, 3])
    ax_img.set_title("adapt 0.1")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1
