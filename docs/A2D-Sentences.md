## A2D-Sentences


### Inference & Evaluation

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 2 --resume [/path/to/model_weight] --backbone [backbone]  --eval
```

For example, evaluating the Video-Swin-Tiny model, run the following command:

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 2 --resume a2d_video_swin_tiny.pth --backbone video_swin_t_p4w7  --eval
```

### Training

- Finetune

```
./scripts/dist_train_a2d.sh [/path/to/output_dir] [/path/to/pretrained_weight] --backbone [backbone]
```

For example, training the Video-Swin-Tiny model, run the following command:
```
./scripts/dist_train_a2d.sh a2d_dirs/video_swin_tiny pretrained_weights/video_swin_tiny_pretrained.pth --backbone video_swin_t_p4w7
```



