## Ref-Youtube-VOS


To evaluate the results, please upload the zip file to the [competition server](https://competitions.codalab.org/competitions/29139#participate-submit_results).


### Inference & Evaluation


First, inference using the trained model.

```
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --output_dir=[/path/to/output_dir] --resume=[/path/to/model_weight] --backbone [backbone] 
```

```
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --output_dir=ytvos_dirs/swin_tiny --resume=ytvos_swin_tiny.pth --backbone swin_t_p4w7
```

If you want to visualize the predited masks, you may add `--visualize` to the above command.

Then, enter the `output_dir`, rename the folder `valid` as `Annotations`. Use the following command to zip the folder:

```
zip -q -r submission.zip Annotations
```

To evaluate the results, please upload the zip file to the [competition server](https://competitions.codalab.org/competitions/29139#participate-submit_results).

### Training


- Finetune 

The following command includes the training and inference stages.

```
./scripts/dist_train_test_ytvos.sh [/path/to/output_dir] [/path/to/pretrained_weight] --backbone [backbone] 
```

For example, training the Video-Swin-Tiny model, run the following command:

```
./scripts/dist_train_test_ytvos.sh ytvos_dirs/video_swin_tiny pretrained_weights/video_swin_tiny_pretrained.pth --backbone video_swin_t_p4w7 
```

- Train from scratch

The following command includes the training and inference stages.

```
./scripts/dist_train_test_ytvos_scratch.sh [/path/to/output_dir] --backbone [backbone] --backbone_pretrained [/path/to/backbone_pretrained_weight] [other args]
```

For example, training the Video-Swin-Tiny model, run the following command:

```
./scripts/dist_train_test_ytvos.sh ytvos_dirs/video_swin_tiny_scratch --backbone video_swin_t_p4w7 --backbone_pretrained video_swin_pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth
```
