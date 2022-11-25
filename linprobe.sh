#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.228.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE

IMAGENET_DIR=./dataset/ILSVRC2012/

# 1 for four node, 4 for single node
ACCUM_ITER=4
PRETRAIN_CHKPT='mae_pretrain_vit_base.pdparams'
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

#export CUDA_VISIBLE_DEVICES=0
#python -m paddle.distributed.launch \
#    --nnodes=$PADDLE_NNODES \
#    --master=$PADDLE_MASTER \
#    --devices=$CUDA_VISIBLE_DEVICES \
#    main_linprobe.py --eval \
#    --resume output_dir/checkpoint-88.pd \
#    --model vit_base_patch16 \
#    --batch_size 512 \
#    --data_path ${IMAGENET_DIR}
