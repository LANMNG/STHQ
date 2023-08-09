#python inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --output_dir ytvos_dirs/video_swin_tiny_scratch --resume ytvos_dirs/video_swin_tiny_scratch/checkpoint.pth  --backbone video_swin_t_p4w7
# ./scripts/dist_test_davis.sh davis_dirs/video_swin_tiny_finetune ytvos_dirs/video_swin_tiny_finetune/checkpoint.pth --backbone video_swin_t_p4w7
#./scripts/dist_test_davis.sh davis_dirs/res50_pretrain coco_dirs/res50_pretrain/checkpoint.pth --backbone resnet50
