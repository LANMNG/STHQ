#./scripts/dist_train_a2d.sh a2d_dirs/video_swin_base_finetune coco_dirs/video_swin_b_pretrain/checkpoint.pth --backbone video_swin_b_p4w7
./scripts/dist_train_test_ytvos.sh ytvos_dirs/video_swin_tiny_finetune coco_dirs/video_swin_t_pretrain/checkpoint.pth --backbone video_swin_t_p4w7
#./scripts/dist_train_test_ytvos.sh ytvos_dirs/res50_finetune coco_dirs/res50_pretrain/checkpoint.pth --backbone resnet50
#./scripts/dist_train_test_ytvos_scratch.sh ytvos_dirs/video_swin_base_scratch --backbone video_swin_b_p4w7 --backbone_pretrained weights/swin_base_patch244_window877_kinetics600_22k.pth
