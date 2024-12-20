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


- [SMOD dataset](https://github.com/bigD233/AMFD.git): SJTU Multispectral Object Detection (SMOD). Within this dataset, 8042 pedestrians, 10478 riders, 6501 bicycles, and 6422 cars are annotated. The degree of occlusion of all objects is meticulously annotated. The dataset with low sampling rate has dense rider and pedestrian objects and contains rich illumination variations in its 3298 pairs of images of night scenarios.



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

- Bispectral Pedestrian Detection Augmented with Saliency Maps using Transformer, VISIGRAPP2022, Mohamed Amine Marnissi et al. [[PDF](https://pdfs.semanticscholar.org/bd9f/468d9f8c6b724ebb369eaf69a8c979f15209.pdf)]

- Attention-Guided Multi-modal and Multi-scale Fusion for Multispectral Pedestrian Detection, PRCV2022, Wei Bao et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-031-18907-4_30)]

- Improving Rgb-Infrared Pedestrian Detection by Reducing Cross-Modality Redundancy, ICIP2022, Qingwang Wang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9897682)]

- Attention-Based Cross-Modality Feature Complementation for Multispectral Pedestrian Detection, IEEE Access, Qunyan Jiang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9775119)]

- DMFFNet: Dual-Mode Multi-Scale Feature Fusion-Based Pedestrian Detection Method,  IEEE Access, Ruizhe Hu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9805743)]

- LGADet: Light-weight Anchor-free Multispectral Pedestrian Detection with Mixed Local and Global Attention, Neural Processing Letters, Xin Zuo et al. [[PDF](https://link.springer.com/article/10.1007/s11063-022-10991-7)]

- Locality guided cross-modal feature aggregation and pixel-level fusion for multispectral pedestrian detection, Information Fusion, Yanpeng Cao et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253522000549)]

- BAANet: Learning Bi-directional Adaptive Attention Gates for Multispectral Pedestrian Detection, ICRA2022, Xiaoxiao Yang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9811999)]

- RGB-Thermal based Pedestrian Detection with Single-Modal Augmentation and ROI Pooling Multiscale Fusion, IGARSS2022, Jiajun Xiang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9883131)]

- MPDFF: Multi-source Pedestrian detection based on Feature Fusion, IGARSS2022, Lingxuan Meng et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9884864)]

- Modality-Independent Regression and Training for Improving Multispectral Pedestrian Detection, ICIVC2022, Han Ni et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9887331)]

- Learning a Dynamic Cross-Modal Network for Multispectral Pedestrian Detection, ACM MM2022, Jin Xie et al. [[PDF](https://dl.acm.org/doi/abs/10.1145/3503161.3547895)]

- Multimodal Object Detection via Probabilistic Ensembling, ECCV2022(oral), Yi-Ting Chen et al. [[PDF](https://arxiv.org/abs/2104.02904.pdf)] [[Code](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling)]

- Translation, Scale and Rotation Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection, ECCV2022, Yuan Maoxun et al. [[PDF](https://arxiv.org/abs/2209.13801)] 

### 2023
- [survey] RGB-T image analysis technology and application: A survey, Engineering Applications of Artificial Intelligence, Kechen Song et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0952197623001033)] 
- [survey] Visible-infrared cross-modal pedestrian detection: a summary, Qian Bie et al. [[PDF](http://www.cjig.cn/jig/ch/reader/view_abstract.aspx?file_no=220670&flag=1)] 
- HAFNet: Hierarchical Attentive Fusion Network for Multispectral Pedestrian Detection,  Remote Sensing, Peiran Peng et al. [[PDF](https://www.mdpi.com/2072-4292/15/8/2041)] 
- Local Adaptive Illumination-Driven Input-Level Fusion for Infrared and Visible Object Detection,  Remote Sensing, Jiawen Wu et al. [[PDF](https://www.mdpi.com/2072-4292/15/3/660)] 
- Multiscale Cross-modal Homogeneity Enhancement and Confidence-aware Fusion for Multispectral Pedestrian Detection, IEEE Transactions on Multimedia, Ruimin Li et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10114594)] 
- Transformer fusion and histogram layer multispectral pedestrian detection network,  Signal, Image and Video Processing, Ying Zang et al. [[PDF](https://link.springer.com/article/10.1007/s11760-023-02579-y)]  
- DaCFN: divide-and-conquer fusion network for RGB-T object detection, International Journal of Machine Learning and Cybernetics, Bofan Wang et al. [[PDF](https://link.springer.com/article/10.1007/s13042-022-01771-9)]  
- Cross-modality complementary information fusion for multispectral pedestrian detection, Neural Computing and Applications, Chaoqi Yan et al. [[PDF](https://link.springer.com/article/10.1007/s00521-023-08239-z)] 
- IGT: Illumination-guided RGB-T object detection with transformers, Knowledge-Based Systems, Keyu Chen et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705123001739)]  
- Learning to measure infrared properties of street views from visible images, Measurement, Lei Wang et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0263224122015160)]  
- Multispectral Pedestrian Detection via Reference Box Constrained CrossAttention and Modality Balanced Optimization, Yinghui Xing et al. [[PDF](https://arxiv.org/pdf/2302.00290.pdf)]  
- Cascaded information enhancement andcross-modal attention feature fusion formultispectral pedestrian detection, Yang Yang et al. [[PDF](https://arxiv.org/pdf/2302.08670.pdf)]  
- Cross-Modality Attention and Multimodal Fusion Transformer for Pedestrian Detection, ECCV 2022 Workshops, Wei-Yu Lee et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-031-25072-9_41)]  
- REVISITING MODALITY IMBALANCE IN MULTIMODAL PEDESTRIAN DETECTION, Arindam Das et al. [[PDF](https://arxiv.org/pdf/2302.12589.pdf)]  
- Illumination-Guided RGBT Object Detection With Inter- and Intra-Modality Fusion, IEEE Transactions on Instrumentation and Measurement, Yan Zhang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10057437)]  
- MCANet: Multiscale Cross-Modality Attention Network for Multispectral Pedestrian Detection,  MultiMedia Modeling, Xiaotian Wang et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-031-27077-2_4)]  
- Multi-modal pedestrian detection with misalignment based on modal-wise regression and multi-modal IoU, Journal of Electronic Imaging, Napat Wanchaitanawong et al. [[PDF](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-32/issue-1/013025/Multi-modal-pedestrian-detection-with-misalignment-based-on-modal-wise/10.1117/1.JEI.32.1.013025.full?SSO=1)]  
- [survey] Low-light Pedestrian Detection in Visible and Infrared Image Feeds: Issues and Challenges, Thangarajah Akilan et al. [[PDF](https://arxiv.org/abs/2311.08557)] 


### 2024
**Survey**
- Surveying You Only Look Once (YOLO) Multispectral Object Detection Advancements, Applications And Challenges, James E. Gallagher et al. [[PDF](https://arxiv.org/abs/2409.12977)]

**Modality Bias**
- Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection, CVPR 2024, Taeheon Kim et al.  [[PDF](https://arxiv.org/pdf/2403.01300.pdf))] [[code](https://github.com/ssbin0914/Causal-Mode-Multiplexer)]  

**Misalignment**
- Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection, CVPR 2024, Chen Chen, et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.pdf)]
- CF-Deformable DETR: An End-to-End Alignment-Free Model for Weakly Aligned Visible-Infrared Object Detection IJCAI 2024, Haolong Fu, et al.  [[PDF](https://www.ijcai.org/proceedings/2024/0084.pdf)] [[code](https://github.com/116508/CF-Deformable-DETR)]  


**Modality Translation**

- HalluciDet: Hallucinating RGB Modality for Person Detection Through Privileged Information, WACV 2024, Medeiros, Heitor Rapela, et al. [[PDF](https://openaccess.thecvf.com/content/WACV2024/html/Medeiros_HalluciDet_Hallucinating_RGB_Modality_for_Person_Detection_Through_Privileged_Information_WACV_2024_paper.html)] [[code](https://github.com/heitorrapela/HalluciDet)]


- Modality Translation for Object Detection Adaptation Without Forgetting Prior Knowledge, ECCV 2024, Medeiros, Heitor Rapela, et al.  [[PDF](https://arxiv.org/pdf/2404.01492)] [[code](https://github.com/heitorrapela/ModTr)]

- TIRDet: Mono-Modality Thermal InfraRed Object Detection Based on Prior Thermal-To-Visible Translation, ACMM, Zeyu Wang, et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3613849)] [[code](https://github.com/zeyuwang-zju/TIRDet)]


**Generalist Model**
- When Pedestrian Detection Meets Multi-Modal Learning: Generalist Model and Benchmark Dataset, ECCV 2024, Yi Zhang et al.  [[PDF](https://arxiv.org/abs/2407.10125)] [[code](https://github.com/BubblyYi/MMPedestron)]

- UniRGB-IR: A Unified Framework for RGB-Infrared Semantic Tasks via Adapter Tuning, Maoxun Yuan et al.  [[PDF](https://arxiv.org/abs/2404.17360)] [[code](https://github.com/PoTsui99/UniRGB-IR)]


**Fusion Architecture**
- Fusion-Mamba for Cross-modality Object Detection, Arxiv, Wenhao Dong et al. [[PDF](https://arxiv.org/pdf/2404.09146)]


- MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection, Arxiv, Xiangbo Gao et al. [[PDF](https://arxiv.org/pdf/2408.01037)] [[code](https://github.com/XiangboGaoBarry/MambaST)]

- ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection, Pattern Recognition, Jifeng Shen et al. [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320323006118)] [[code](https://github.com/chanchanchan97/ICAFusion)]

- CrossFormer: : Cross-guided attention for multi-modal object detection, Pattern Recognition Letters, Seungik Lee et al. [[PDF](https://dl.acm.org/doi/10.1016/j.patrec.2024.02.012)] 

- Damsdet: Dynamic adaptive multispectral detection transformer with competitive query selection and adaptive feature fusion, ECCV 2024, Junjie Guo et al.  [[PDF](https://arxiv.org/pdf/2403.00326)] [[code](https://github.com/gjj45/DAMSDet)]



- MiPa: Mixed Patch Infrared-Visible Modality Agnostic Object Detection, ARXIV 2024, Medeiros, Heitor Rapela, et al.  [[PDF](https://arxiv.org/abs/2404.18849)] [[code](https://github.com/heitorrapela/MiPa)]  

- Removal then Selection: A Coarse-to-Fine Fusion Perspective for RGB-Infrared Object Detection, Arxiv, Tianyi Zhao et al. [[PDF](https://arxiv.org/pdf/2401.10731)] [[code](https://github.com/Zhao-Tian-yi/RSDet)]


-  Rethinking Early-Fusion Strategies for Improved Multispectral Object Detection, IEEE Transactions on Intelligent Vehicles, 2024, Xue Zhang, et al.  [[PDF](https://ieeexplore.ieee.org/document/10681477)] [[code](https://github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection)] 


- RGB-X Object Detection via Scene-Specific Fusion Modules, WACV 2024, Sri Aditya Deevi, et al.  [[PDF](https://github.com/dsriaditya999/RGBXFusion)][[code](https://github.com/dsriaditya999/RGBXFusion)] 

- Rethinking Self-Attention for Multispectral Object Detection, IEEE Transactions on Intelligent Transportation Systems, 2024, Sijie Hu, et al.  [[PDF](https://ieeexplore.ieee.org/document/10565297)][[code](https://github.com/Superjie13/CPCF_Multispectral)] 

- TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection, 2024, Xue Zhang, et al.  [[PDF](https://ieeexplore.ieee.org/document/10645696)][[code](https://github.com/xuez-phd/tfdet)] 

- Multidimensional Fusion Network for Multispectral Object Detection, IEEE Transactions on Circuits and Systems for Video Technology, 2024, Fan Yang, et al.  [[PDF](https://ieeexplore.ieee.org/document/10666754)]

- FoRA: Low-Rank Adaptation Model beyond Multimodal Siamese Network,Arxiv, Weiying Xie, et al.  [[PDF](https://arxiv.org/abs/2407.16129)]


**DETR**
- GM-DETR: Generalized Muiltispectral DEtection TRansformer with Efficient Fusion Encoder for Visible-Infrared Detection, Yiming Xiao, et al.  [[PDF](https://openaccess.thecvf.com/content/CVPR2024W/JRDB/papers/Xiao_GM-DETR_Generalized_Muiltispectral_DEtection_TRansformer_with_Efficient_Fusion_Encoder_for_CVPRW_2024_paper.pdf)][[code](https://github.com/yiming-shaw/GM-DETR)] 

- DPDETR: Decoupled Position Detection Transformer for Infrared-Visible Object Detection, 
Junjie Guo, et al.  [[PDF](https://arxiv.org/abs/2408.06123)]

- MS-DETR: Multispectral Pedestrian Detection Transformer with Loosely Coupled Fusion and Modality-Balanced Optimization, IEEE Transactions on Intelligent Transportation Systems 2024, Yinghui Xing, et al.  [[PDF](https://arxiv.org/abs/2302.00290v1)] [[code](https://github.com/YinghuiXing/MS-DETR)]  

**New Dataset**
- AMFD: Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection, Arxiv, Zizhao Chen, et al.  [[PDF](https://arxiv.org/abs/2405.12944)] [[code](https://github.com/bigD233/AMFD)]  
  
**Few Shot**
- Cross-modality interaction for few-shot multispectral object detection with semantic knowledge, Neural Networks, 2024, Lian Huang, et al.  [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0893608024000807)]


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

