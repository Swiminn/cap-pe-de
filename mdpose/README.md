MDPose: PyTorch Implementation
===========================================================

This repository is the PyTorch implementation of  
"[MDPose: Real-Time Multi-Person Pose Estimation via Mixture Density Model](https://arxiv.org/abs/2302.08751)"

Environment
-----------
- python 3.8.19
- pytorch 2.3.1
- torchvision 0.18.1
- numpy 1.23.4
- Pillow 10.4.0
- scipy 1.10.1
- opencv-python 4.5.5

Directory Structure
-------------------
```
(root-directory)
├── README.md
├── run_mmpe_coco.py
├── src
│   └── (python-source-file.py)
├── result
│   └── (result-directory)
│       ├── ...
│       └── snapshot
│           └── (iteration)
│               ├── network.pth
│               └── optimizer.pth
└── data
    └── coco-2017
        ├── annotations
        └── images
```
You can download the voc and coco dataset in the follow links.  
http://cocodataset.org/#download (coco-2017)

Usage
-----
Training
```
# run_mmpe_coco.sh
--training_args="{'max_iter': maximum number of iterations, ...}"

# command
.../(root-directory)$ bash run_mmpe_coco.sh
```

Much of this repository references [Mixture-Model-based Object Detector](https://arxiv.org/abs/1911.12721) repo.

Citation
--------
```
@inproceedings{seo2023mdpose,
  title={MDPose: real-time multi-person pose estimation via mixture density model},
  author={Seo, Seunghyeon and Yoo, Jaeyoung and Hwang, Jihye and Kwak, Nojun},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1868--1878},
  year={2023},
  organization={PMLR}
}
```
