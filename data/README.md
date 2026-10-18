# Data

Place the preprocessed datasets in this directory (one sub-directory per city):

```
data/
  Porto/all/    train.csv  val.csv  test.csv  road_id_to_index.json
                grid_description.json  grid_feature_map.npy  road_info.csv
                adjacency_matrix.npy
  Xiamen/all/   (same structure)
  Shanghai/all/ (same structure; downloaded separately, see README)
```

Notes:
- The Porto and Xiamen datasets are small enough to be committed directly.
  Large dense arrays (e.g., `adjacency_matrix.npy`) should either be stored via
  [Git LFS](https://git-lfs.com/) or committed in sparse format.
- The Shanghai dataset is large; download it separately (see the link in the
  top-level README).
