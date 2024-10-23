#CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=23456 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml

# set -x

# GPUS_PER_NODE=${GPUS_PER_NODE:-2}
# NNODES=${NNODES:-1}
# NODE_RANK=0
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=12320

# DISTRIBUTED_ARGS="
# --nproc_per_node $GPUS_PER_NODE
# --nnodes $NNODES
# --node_rank $NODE_RANK
# --master_addr $MASTER_ADDR
# --master_port $MASTER_PORT
# "

#CUDA_VISIBLE_DEVICES=6,7 torchrun $DISTRIBUTED_ARGS train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml

CUDA_VISIBLE_DEVICES=2 python3 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml