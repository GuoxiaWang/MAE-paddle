## [Masked Autoencoders](https://github.com/facebookresearch/mae): A PaddlePaddle Re-Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a PaddlePaddle/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)


### Fine-tuning

```
#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.123.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE

# batch_size 32, ACCUM_ITER=4, effective batch size: 1024
# batch_size 128, ACCUM_ITER=1, effective batch size: 1024
ACCUM_ITER=1
PRETRAIN_CHKPT='mae_pretrain_vit_base.pdparams'
IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    main_finetune.py \
    --accum_iter $ACCUM_ITER \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

|                        | ViT-Base | ViT-Large | ViT-Huge |
| ---------------------- | -------- | --------- | -------- |
| official (PyTorch/GPU) | 83.664   | 85.952    | 86.928   |
| this repo (Paddle/GPU) | 83.568   | -         | -        |

### Linear Probing

```
#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.123.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE


# batch_size 512, ACCUM_ITER=4, effective batch size: 16384

# 1 for four node, 4 for single node
ACCUM_ITER=4
PRETRAIN_CHKPT='mae_pretrain_vit_base.pdparams'
IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
   --nnodes=$PADDLE_NNODES \
   --master=$PADDLE_MASTER \
   --devices=$CUDA_VISIBLE_DEVICES \
   main_linprobe.py \
   --accum_iter $ACCUM_ITER \
   --batch_size 512 \
   --model vit_base_patch16 \
   --cls_token \
   --finetune ${PRETRAIN_CHKPT} \
   --epochs 90 \
   --blr 0.1 \
   --weight_decay 0.0 \
   --dist_eval --data_path ${IMAGENET_DIR}
```

|                        | ViT-Base | ViT-Large | ViT-Huge |
| ---------------------- | -------- | --------- | -------- |
| official (PyTorch/GPU) | 67.8     | 76.0      | 77.2     |
| this repo (Paddle/GPU) | 67.7     | -         | -        |

```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
