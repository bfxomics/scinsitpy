import anndata as an
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mplPolygon

from scinsitpy.io.basic import get_palette, load_bounds_pixel


def scinsit_crop(
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
    Return the Anndata object of the crop region.
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


def scinsit_qc(adata: an.Anndata, library_id: str) -> int:
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


def scinsit_compartment(
    adata: an.Anndata, celltypelabel: str = "celltype", metalabel: str = "population", figsize: tuple = (20, 5)
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
