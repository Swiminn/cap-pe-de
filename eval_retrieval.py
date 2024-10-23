# /home/user/kyomin/LAVIS/lavis/models/blip2_models/blip2.py
# /home/user/kyomin/LAVIS/lavis/models/blip_models/blip_retrieval.py

import argparse
import pickle
import os
import json

import numpy as np

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.utils import now
from lavis.runners.runner_base import RunnerBase


def get_score_matrix(file_dir):

    i2t_file = os.path.join(file_dir, "i2t.p")
    t2i_file = os.path.join(file_dir, "t2i.p")

    with open (i2t_file, 'rb') as f:
        i2t_score = pickle.load(f)
    with open (t2i_file, 'rb') as f:
        t2i_score = pickle.load(f)

    return i2t_score.cpu(), t2i_score.cpu()


def report_metrics(scores_i2t, scores_t2i, txt2img, img2txt, out_file):
    # Images->Text
    # import pdb; pdb.set_trace()
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        # idx = np.argsort(score)[::-1]
        idx = np.argsort(score).flip(0)
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(idx == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        # idx = np.argsort(score)[::-1]
        idx = np.argsort(score).flip(0)
        ranks[index] = np.where(idx == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    agg_metrics = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
        "agg_metrics": agg_metrics,
    }
    with open(out_file, "a") as f:
        f.write(json.dumps(eval_result) + "\n")

    return eval_result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Retrieval evaluation")

    parser.add_argument("--cfg_path", type=str, help="path to config file")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    parser.add_argument("--score_file", type=str, help="timestamp of the score matrix file")
    parser.add_argument("--out_dir", type=str, help="evaluation output directory")

    args = parser.parse_args()

    score_path = args.score_file
    out_path = args.out_dir
    out_file = os.path.join(out_path, "evaluate.json")

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    cfg = Config(args)
    job_id = now()
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    datasets = task.build_datasets(cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    # for split_name in datasets:
    split_name = "test"
    test_loader = runner.dataloaders.get(split_name, None)
    assert test_loader, "test_loader for split {} is None.".format(split_name)

    # import pdb; pdb.set_trace()
    txt2img = test_loader.loader.dataset.txt2img
    img2txt = test_loader.loader.dataset.img2txt

    i2t_score_matrix, t2i_score_matrix = get_score_matrix(score_path)
    evaluation = report_metrics(i2t_score_matrix, t2i_score_matrix, txt2img, img2txt , out_file)

    print(f"Evaluation results: {evaluation}")
    print(f"Evaluation result stored as json file at: {out_file}")
