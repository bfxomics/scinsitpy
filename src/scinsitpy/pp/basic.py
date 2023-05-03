import anndata as an
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt


def filter_and_run_scanpy(adata: an.AnnData, min_counts: int) -> an.AnnData:
    """Filter and run scanpy analysis.

    Parameters
    ----------
    adata
        Anndata object.
    min_counts
        minimum transcript count to keep cell.

    Returns
    -------
    Anndata analyzed object.
    """
    sc.pp.filter_cells(adata, min_counts=min_counts, inplace=True)

    print("total cells=", adata.shape[0])
    print("mean transcripts per cell=", adata.obs["barcodeCount"].mean())
    print("median transcripts per cell=", adata.obs["barcodeCount"].median())

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="clusters")

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(adata, "umap", color="clusters", ax=axs[0], show=False)
    sns.scatterplot(x="x_pix", y="y_pix", data=adata.obs, s=2, hue="clusters")

    return adata
