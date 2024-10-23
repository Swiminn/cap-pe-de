import os
import numpy as np
import sys
sys.path.append('/home/user/kyomin/LAVIS/pycocoevalcap')
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

def coco_caption_eval(results_file, annotation_file=None):
        
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    coco_eval.evaluate()
    BLEU_score = np.exp((np.log(coco_eval.eval["Bleu_1"]) + np.log(coco_eval.eval["Bleu_2"]) + np.log(coco_eval.eval["Bleu_3"]) + np.log(coco_eval.eval["Bleu_4"]))/4)
    print(f"computed BLEU scores")
    print(f"BLEU_1: {coco_eval.eval['Bleu_1']*100:.1f}")
    print(f"BLEU_2: {coco_eval.eval['Bleu_2']*100:.1f}")
    print(f"BLEU_3: {coco_eval.eval['Bleu_3']*100:.1f}")
    print(f"BLEU_4: {coco_eval.eval['Bleu_4']*100:.1f}")
    print(f"BLEU: {BLEU_score*100:.1f}")

    return coco_eval

if __name__=="__main__":

    coco_gt_root = "/home/user/kyomin/LAVIS/lavis/coco_gt"
    filenames = {
            "val": "coco_karpathy_val_gt_en.json",
            "test": "coco_karpathy_test_gt_en.json",
        }
    
    results_file = "/home/user/kyomin/LAVIS/lavis/output/BLIP2/Caption_coco_opt2.7b/20240905001/result/test_epochbest.json"
    
    annotation_file = os.path.join(coco_gt_root, filenames["test"])

    coco_caption_eval(results_file, annotation_file=annotation_file)