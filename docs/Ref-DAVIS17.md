## Ref-DAVIS17


### Inference & Evaluation

```
./scripts/dist_test_davis.sh [/path/to/output_dir] [/path/to/model_weight] --backbone [backbone]
```

For example, evaluating the Swin-Large model, run the following command:

```
./scripts/dist_test_davis.sh davis_dirs/swin_large ytvos_swin_large.pth --backbone swin_l_p4w7
```