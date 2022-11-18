#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.228.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE

PRETRAIN_CHKPT=''
#    --finetune ${PRETRAIN_CHKPT} \
IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    main_linprobe.py \
    --batch_size 512 \
    --model vit_large_patch16 \
    --cls_token \
    --epochs 90 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
