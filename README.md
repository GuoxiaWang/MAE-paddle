## [Masked Autoencoders](https://github.com/facebookresearch/mae): A PaddlePaddle Re-Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a PaddlePaddle/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):

### Linear Probing

```
#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.123.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE


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

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">official (PyTorch/GPU)</td>
<td align="center">67.8</td>
<td align="center">76.0</td>
<td align="center">77.2</td>
</tr>
<tr><td align="left">this repo (Paddle/GPU)</td>
<td align="center">67.7</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
</tbody></table>

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
