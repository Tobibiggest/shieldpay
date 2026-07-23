# Installing PyTorch Geometric

`torch` and `torch-geometric` install fine from `requirements.txt` on any OS via plain
`pip`. PyG's *accelerated extension packages* — `torch_scatter`, `torch_sparse`,
`torch_cluster`, `torch_spline_conv` — do not, because their wheels are built per
combination of OS, Python version, PyTorch version, and CUDA version (or CPU-only).
Installing them via a plain `pip install` line in `requirements.txt` would silently
break on any machine with a different combination than the one it was written for.

Core model code in this package (`HGTConv`, `SAGEConv`, `GATConv`, `HeteroConv`,
message passing) works without these extensions — PyG falls back to pure-PyTorch
scatter/sparse ops, just slower. Install them only if you need the speedup (large
graphs, GPU training).

## Steps

1. Install base requirements first:
   ```bash
   pip install -r requirements.txt
   ```
2. Check your installed torch version and whether you have CUDA:
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```
3. Install the matching extension wheels from PyG's wheel index. Replace
   `{TORCH_VERSION}` and `{cpu|cuXXX}` with your actual values from step 2
   (e.g. `2.2.0` and `cpu`, or `2.2.0` and `cu121`):
   ```bash
   pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
       -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{cpu|cuXXX}.html
   ```
4. Verify:
   ```bash
   python -c "import torch_geometric; from torch_geometric.nn import HGTConv, SAGEConv, GATConv; print('ok')"
   ```

If you skip steps 2-4, everything still runs — just without the compiled scatter/sparse
kernels.
