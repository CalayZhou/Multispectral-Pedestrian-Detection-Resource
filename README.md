---
# Multispectral Pedestrian Detection Resource
A list of resouces for multispectral pedestrian detection,including the datasets, methods, annotations and tools.

---
## Datasets
- [KAIST dataset](https://soonminhwang.github.io/rgbt-ped-detection/): The KAIST Multispectral Pedestrian Dataset consists of 95k color-thermal pairs (640x480, 20Hz) taken from a vehicle. All the pairs are manually annotated (person, people, cyclist) for the total of 103,128 dense annotations and 1,182 unique pedestrians. The annotation includes temporal correspondence between bounding boxes like Caltech Pedestrian Dataset. 
- [CVC-14 dataset](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/): The CVC-14 dataset is composed by two sets of sequences. These sequences are named as the day and night sets,  which refers to the moment of the day they were acquired, and Visible and FIR depending the camera that was user to recor the sequences. For training 3695 images during the day, and 3390 images during night, with around 1500 mandatory pedestrian annotated for each sequence. For testing around 700 images for both sequences with around 2000 pedestrian during day, and around 1500 pedestrian during night.
- [FLIR dataset](https://www.flir.cn/oem/adas/adas-dataset-form/): Synced annotated thermal imagery and non-annotated RGB imagery for reference. It should to noted that the infrared and RGB images are not aligned. The FLIR dataset has 10,228 total frames and 9,214 frames with bounding boxes.(28151 Person, 46692 Car, 4457 Bicycle, 240 Dog, 2228 Other Vehicle)

---
## Methods

- Multispectral Pedestrian Detection Benchmark Dataset and Baseline, 2015, Soonmin Hwang et al.
[[PDF](https://soonminhwang.github.io/rgbt-ped-detection/misc/CVPR15_Pedestrian_Benchmark.pdf)]
[[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

- Multispectral Deep Neural Networks for Pedestrian Detection, 2016, Jingjing Liu et al.
[[PDF](https://arxiv.org/abs/1611.02644)]
[[Code](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn)]

- Multi-spectral Pedestrian Detection Based on Accumulated Object Proposal with Fully Convolutional Networks, 2016, Hangil Choi et al.
[[PDF](https://ieeexplore.ieee.org/document/7899703)]

- Fully Convolutional Region Proposal Networks for Multispectral Person Detection, 2017, Daniel König et al.
[[PDF](https://ieeexplore.ieee.org/abstract/document/8014770)]

- Unified Multi-spectral Pedestrian Detection Based on Probabilistic Fusion Networks, 2017, Kihong Park et al.
[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300906)]

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


---
## Improved KAIST Annotations
 - Improved KAIST Testing Annotations provided by Liu et al.[Link to download](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) 
 - Improved KAIST Training Annotations provided by Zhang et al.[Link to download](https://github.com/luzhang16/AR-CNN) 

 ---
## Tools
- Evalutaion codes.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper)
- Annotation: vbb format->xml format.[Link to download](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data/scripts)

