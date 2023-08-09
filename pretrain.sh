python -m torch.distributed.launch --nproc_per_node=4  --use_env \
main_pretrain.py --dataset_file all --binary --with_box_refine \
--batch_size 4 --num_frames 1 --epochs 8  --lr_drop 4 6 \
--output_dir coco_dirs/video_swin_t_pretrain_q5_w3 --backbone video_swin_t_p4w7 \
--resume coco_dirs/video_swin_t_pretrain_q5_w3/checkpoint.pth
#--backbone_pretrained weights/swin_tiny_patch244_window877_kinetics400_1k.pth \
#--resume coco_dirs/video_swin_t_pretrain_q5_w3/checkpoint.pth
