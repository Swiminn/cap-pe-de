# Integrated Model for Object Detection / Pose Estimation / Captioning

This repository is based on research aimed at integrating three key computer vision tasks—Object Detection, Pose Estimation, and Image Captioning—into a unified framework. By leveraging a Transformer-based backbone, we ensure consistency and efficiency across all three tasks, laying the groundwork for improved performance and easier maintenance.

## Introduction

Humans perceive the world through multiple sensory channels, compensating for a lack of information in one modality by relying on another. Inspired by this cognitive process, recent AI research has focused on integrating diverse data modalities to enhance model capacity and applicability. This project combines three traditionally separate tasks into a single model:

- **Object Detection:** Identifying the locations and categories of objects within images.
- **Pose Estimation:** Predicting the positions of human joints to understand body poses.
- **Captioning:** Generating natural language descriptions of image content.

By unifying these tasks under one framework, we aim to improve efficiency, reduce complexity, and capitalize on the synergy among these tasks.

## Key Features

1. **Transformer-Based Backbone**  
   Instead of relying on traditional CNN backbones, we utilize a Transformer architecture. Self-Attention enables global context modeling and improved handling of complex visual patterns, providing a common feature extraction layer suitable for multiple tasks.

2. **Object Detector**  
   Our object detector is built upon YOLOv5, enhanced with a Transformer backbone. It supports datasets like COCO and K-Fashion to cover a broad range of objects, including general items and specialized categories like fashion apparel.

3. **Pose Estimator (MDPose)**  
   The pose estimator employs a Mixture Density Model-based approach, known as MDPose, allowing for real-time multi-person pose estimation. By integrating a Transformer backbone, we ensure robust performance and accurate joint detection.

4. **Captioning**  
   Leveraging the BLIP2 framework and Low-Rank Adaptors (LㅐRA), our captioning model excels in zero-shot scenarios, effectively describing unseen images. This approach was showcased in the CVPR 2023 NICE Workshop, reflecting its strong generalization capabilities.

5. **Modular Design**  
   Each task is handled by its own specialized head module built atop a shared Transformer backbone. This modularity allows for efficient maintenance, easier troubleshooting, and seamless expansion to additional tasks in the future.

