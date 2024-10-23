export CUDA_VISIBLE_DEVICES=3
python3 eval_retrieval.py \
  --cfg_path lavis/projects/blip2/eval/ret_coco_eval.yaml \
  --score_file lavis/output/BLIP2/Retrieval_COCO/SimMatrix/20240927_1215 \
  --out_dir lavis/output/BLIP2/Retrieval_COCO/quick
