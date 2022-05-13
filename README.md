---
# Multispectral Pedestrian Detection Resource
A list of resouces for multispectral pedestrian detection,including the datasets, methods, annotations, evaluation and tools.

---
## Datasets
- [KAIST dataset](https://soonminhwang.github.io/rgbt-ped-detection/): The KAIST Multispectral Pedestrian Dataset consists of 95k color-thermal pairs (640x480, 20Hz) taken from a vehicle. All the pairs are manually annotated (person, people, cyclist) for the total of 103,128 dense annotations and 1,182 unique pedestrians. The annotation includes temporal correspondence between bounding boxes like Caltech Pedestrian Dataset. 
- [CVC-14 dataset](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/): The CVC-14 dataset is composed by two sets of sequences. These sequences are named as the day and night sets,  which refers to the moment of the day they were acquired, and Visible and FIR depending the camera that was user to recor the sequences. For training 3695 images during the day, and 3390 images during night, with around 1500 mandatory pedestrian annotated for each sequence. For testing around 700 images for both sequences with around 2000 pedestrian during day, and around 1500 pedestrian during night.
- [FLIR dataset](https://www.flir.cn/oem/adas/adas-dataset-form/): Synced annotated thermal imagery and non-annotated RGB imagery for reference. It should to noted that the infrared and RGB images are not aligned. The FLIR dataset has 10,228 total frames and 9,214 frames with bounding boxes(28151 Person, 46692 Car, 4457 Bicycle, 240 Dog, 2228 Other Vehicle).
 
  >In the original FLIR dataset, the thermal and visible images are not aligned. So Heng Zhang et al manually aligned the visible-thermal image pairs and end up with 4128 pairs   for training and 1013 pairs for validation. The aligned version dataset can be downloaded here: https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view (This   aligned dataset was firstly mentioned in Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks, ICIP 2020, Heng Zhang et al.)
 
- [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/): This dataset contains 30976 images, or 15488 pairs, most of which were taken at very dark scenes, and all of the images are strictly aligned in time and space. Pedestrians in the dataset are labeled. We compare the dataset with other visible-infrared datasets and evaluate the performance of some popular visual algorithms including image fusion, pedestrian detection and image-to-image translation on the dataset.

- [Autonomous Vehicles dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/): A novel multispectral dataset was generated for autonomous vehicles that consists of RGB, NIR, MIR, and FIR images, which prepared 7,512 images in total (3,740 taken at daytime and 3,772 taken at nighttime).
---
## Methods

### before 2018
- Multispectral Pedestrian Detection Benchmark Dataset and Baseline, 2015, Soonmin Hwang et al.
[[PDF](https://soonminhwang.github.io/rgbt-ped-detection/misc/CVPR15_Pedestrian_Benchmark.pdf)]
[[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

- Multispectral Pedestrian Detection using Deep Fusion Convolutional Neural Networks, 2016, Jörg Wagner et al.
[[PDF](https://www.researchgate.net/publication/302514661_Multispectral_Pedestrian_Detection_using_Deep_Fusion_Convolutional_Neural_Networks)]

- Multispectral Deep Neural Networks for Pedestrian Detection, 2016, Jingjing Liu et al.
[[PDF](https://arxiv.org/abs/1611.02644)]
[[Code](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn)]

- Multi-spectral Pedestrian Detection Based on Accumulated Object Proposal with Fully Convolutional Networks, 2016, Hangil Choi et al.
[[PDF](https://ieeexplore.ieee.org/document/7899703)]

- Fully Convolutional Region Proposal Networks for Multispectral Person Detection, 2017, Daniel König et al.
[[PDF](https://ieeexplore.ieee.org/abstract/document/8014770)]

- Unified Multi-spectral Pedestrian Detection Based on Probabilistic Fusion Networks, 2017, Kihong Park et al.
[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300906)]

### 2018
- Fusion of Multispectral Data Through Illumination-aware Deep Neural Networks for Pedestrian Detection, 2018, Dayan Guan et al.
[[PDF](https://arxiv.org/abs/1802.09972)]
[[Code](https://github.com/dayanguan/illumination-aware_multispectral_pedestrian_detection/)]

- Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection, BMVC 2018, Chengyang Li et al.
[[PDF](https://arxiv.org/pdf/1802.09972.pdf)]
[[Code](https://github.com/Li-Chengyang/IAF-RCNN)]

- Pedestrian detection at night by using Faster R-CNN infrared images, 2018, Michelle Galarza Bravo et al.
[[PDF](https://ingenius.ups.edu.ec/index.php/ingenius/article/download/20.2018.05/2767)]

- Real-Time Multispectral Pedestrian Detection with a Single-Pass Deep Neural Network, 2018, Maarten Vandersteegen et al.
[[PDF](https://link.springer.com/chapter/10.1007/978-3-319-93000-8_47)]

- Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.
[[PDF](https://arxiv.org/abs/1808.04818)]
[[Code](https://github.com/Li-Chengyang/MSDS-RCNN)]
[[Project Link](https://li-chengyang.github.io/home/MSDS-RCNN/)]

### 2019
- Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pesdestrian Detecion, 2019, Yanpeng Cao et al.
[[PDF](https://arxiv.org/abs/1902.05291)]
[[Code](https://github.com/dayanguan/realtime_multispectral_pedestrian_detection)]

 - Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al.
[[PDF](https://arxiv.org/abs/1901.02645)]
[[Code](https://github.com/luzhang16/AR-CNN)]

- The Cross-Modality Disparity Problem in Multispectral Pedestrian Detection, 2019, Lu Zhang et al.
[[PDF](https://arxiv.org/abs/1901.02645v1)]

- Cross-modality interactive attention network for multispectral pedestrian, 2019, Lu Zhang et al.
[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253518304111)]
[[Code](https://github.com/luzhang16/CIAN)]

- GFD-SSD  Gated Fusion Double SSD for Multispectral Pedestrian Detection, 2019, Yang Zheng et al.
[[PDF](https://arxiv.org/abs/1903.06999)]

-  Unsupervised Domain Adaptation for Multispectral Pedestrian Detection, 2019, Dayan Guan et al.
[[PDF](https://arxiv.org/abs/1904.03692)]
[[Code](https://github.com/dayanguan/unsupervised_multispectral_pedestrian_detectio)]

-  Generalization ability of region proposal networks for multispectral person detection, 2019, Kevin Fritz et al.[[PDF](https://arxiv.org/abs/1905.02758)]

- Borrow from Anywhere: Pseudo Multi-modal Object Detection in Thermal Imagery, 2019, Chaitanya Devaguptapu et al. [[PDF](https://arxiv.org/abs/1905.08789)]

### 2020
- Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks, ICIP 2020, Heng Zhang et al. [[PDF](https://hal.archives-ouvertes.fr/hal-02872132/file/icip2020.pdf)]

- Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]

- Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)]

- Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]

- Robust pedestrian detection in thermal imagery using synthesized images, ICPR 2020, My Kieu et al.[[PDF](https://arxiv.org/abs/2102.02005)]

### 2021

- Pixel Invisibility: Detecting Objects Invisible in Color Image, 2021, Yongxin Wang et al.[[PDF](https://arxiv.org/pdf/2006.08383.pdf)]

- Guided Attentive Feature Fusion for Multispectral Pedestrian Detection, WACV 2021, Heng Zhang et al. [[PDF](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)]

- Deep Active Learning from Multispectral Data Through Cross-Modality Prediction Inconsistency, ICIP2021, Heng Zhang et al. [[PDF](https://hal.archives-ouvertes.fr/hal-03236409/document)]

- Spatio-Contextual Deep Network Based Multimodal Pedestrian Detection For Autonomous Driving, Kinjal Dasgupta et al. [[PDF](https://arxiv.org/abs/2105.12713)]

- Uncertainty-Guided Cross-Modal Learning for Robust Multispectral Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, Jung Uk Kim et al. [[PDF](https://ieeexplore.ieee.org/document/9419080)]

- Cross-Modality Fusion Transformer for Multispectral Object Detection, 2021, Qingyun Fang et al. [[PDF](https://arxiv.org/pdf/2111.00273v2.pdf)]

- Weakly Aligned Feature Fusion for Multimodal Object Detection, 2021, Lu Zhang et al.  [[PDF](https://ieeexplore.ieee.org/abstract/document/9523596)]

- Attention Fusion for One-Stage Multispectral Pedestrian Detection, 2021, Zhiwei Cao et al. [[PDF](https://www.mdpi.com/1424-8220/21/12/4184)]

- Multi-Modal Pedestrian Detection with Large Misalignment Based on Modal-Wise Regression and Multi-Modal IoU, 2021, Napat Wanchaitanawong et al. [[PDF](https://arxiv.org/pdf/2107.11196.pdf)]

- MLPD: Multi-Label Pedestrian Detector in Multispectral Domain, 2021, Jiwon Kim et al. [[PDF](https://ieeexplore.ieee.org/document/9496129)]

- [survey] From handcrafted to deep features for pedestrian detection: a survey, IEEE Transactions on Pattern Analysis and Machine Intelligence 2021, Jiale Cao et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9420291)]

### 2022
- Low-Cost Multispectral Scene Analysis With Modality Distillation, WACV 2022, Heng Zhang et al. [[PDF](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_Low-Cost_Multispectral_Scene_Analysis_With_Modality_Distillation_WACV_2022_paper.html)]

- Confidence-aware Fusion using Dempster-Shafer Theory for Multispectral Pedestrian Detection, IEEE Transactions on Multimedia 2022, Qing Li et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9739079)]

- PIAFusion: A progressive infrared and visible image fusion network based on illumination aware, Information Fusion, Linfeng Tang et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S156625352200032X)]  [[Code](https://github.com/Linfeng-Tang/PIAFusion)]

- Improving RGB-Infrared Object Detection by Reducing Cross-Modality Redundancy, Remote Sensing, Qingwang Wang et al. [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]

- Spatio-contextual deep network-based multimodal pedestrian detection for autonomous driving, IEEE Transactions on Intelligent Transportation Systems, Kinjal Dasgupta et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9706418)]

- Robust Thermal Infrared Pedestrian Detection By Associating Visible Pedestrian Knowledge, ICASSP 2022, Sungjune Park et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9746886)]

- Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning, IEEE Transactions on Circuits and Systems for Video Technology, Yiming Sun. [[PDF](https://ieeexplore.ieee.org/abstract/document/9759286)] [[Code](https://github.com/VisDrone/DroneVehicle)]

- Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI2022, Jung Uk Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]

- Bispectral Pedestrian Detection Augmented with Saliency Maps using Transformer, VISIGRAPP 2022, Mohamed Amine Marnissi et al. [[PDF](https://pdfs.semanticscholar.org/bd9f/468d9f8c6b724ebb369eaf69a8c979f15209.pdf)]

---
## Improved KAIST Annotations
 - Improved KAIST Testing Annotations provided by Liu et al.[Link to download](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) 
 - Improved KAIST Training Annotations provided by Zhang et al.[Link to download](https://github.com/luzhang16/AR-CNN) 

## [Evaluation Criteria](https://eval.ai/web/challenges/challenge-page/1247/evaluation)  
- Training Annotations: The KAIST Multispectral Pedestrian Dataset has three kinds of annotations for training. First, the original annotations were provided by Hwang et al. [1]. Second, the sanitized annotations were provided by Li et al. [2]. Lastly, the paired annotations were provided by Zhang et al. [3].

- Test Annotations: The sanitized annotations [2] are used in this challenge for evaluation. The sanitized annotations eliminate the annotation errors, including imprecise localization, misclassification and misaligned regions. The annotations are mostly used in recent works for evaluation, and therefore we also adopt the annotations to conduct a fair comparison.

  [1] - S. Hwang, J. Park, N. Kim, Y. Choi, and I. Kweon, “Multispectral pedestrian detection: Benchmark dataset and baseline,” in Proc. IEEE Conf. Comput. Vision Pattern Recognit., 2015, pp. 1037–1045
  
  [2] - C. Li, D. Song, R. Tong, and M. Tang, “Multispectral pedestrian detection via simultaneous detection and segmentation,” in Proc. Brit. Mach. Vision Conf., 2018, pp. 225.1–225.12.
  
  [3] - L. Zhang, X. Zhu, X. Chen, X. Yang, Z. Lei, and Z. Liu, “Weakly aligned cross-modal learning for multispectral pedestrian detection,” in Proc. IEEE Int. Conf. Comput. Vision, 2019, pp. 5126–5136.
  
 ---
## Tools
- Evalutaion codes.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper)
- Annotation: vbb format->xml format.[Link to download](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data/scripts)

