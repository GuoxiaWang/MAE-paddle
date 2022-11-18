#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.228.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE

PRETRAIN_CHKPT='xxxxx'
#    --finetune ${PRETRAIN_CHKPT} \
IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    main_finetune.py \
    --batch_size 32 \
    --model vit_base_patch16 \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}



#export CUDA_VISIBLE_DEVICES=0
#python -m paddle.distributed.launch \
#    --nnodes=$PADDLE_NNODES \
#    --master=$PADDLE_MASTER \
#    --devices=$CUDA_VISIBLE_DEVICES \
#    main_finetune.py --eval \
#    --resume mae_finetuned_vit_base.pdparams \
#    --model vit_base_patch16 \
#    --batch_size 16 \
#    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#    --data_path ${IMAGENET_DIR}
