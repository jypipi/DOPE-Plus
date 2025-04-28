[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)

# Deep Object Pose Estimation

This is the public repository for our DOPE-Plus project


## Contents
This repository contains our source code for [training](train), [inference](inference), and [synthetic data generation](data_generation) using [Blenderproc](https://github.com/DLR-RM/BlenderProc), which were built upon the original DOPE codebase.


## Tested Configurations

We have tested our standalone training and inference scripts on Ubuntu 20.04 with Python 3.8.10, using NVIDIA RTX 4090 and A2000 GPUs. 


## Datasets

We have trained and tested DOPE with two publicaly available datasets: YCB, and HOPE.
We trained and tested DOPE-Plus with our HOPE-Syn&Real Dataset and the Synthetic Block Dataset. All synthetic images were generated with our enhanced data generation pipeline. The HOPE-Syn&Real Dataset also contains real images from the publicaly available dataset HOPE.


### 3D Textured Models
The [HOPE dataset](https://github.com/swtyree/hope-dataset/) is a collection of RGBD images and video sequences with labeled 6-DoF poses for 28 toy grocery objects.  The 3D models [can be  downloaded here](https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft). 

In addition, we included our [3D texured model of the Block](data_generation/blenderproc_data_gen/models/Block_w_sandpaper_obj/).

<br><br>

---


## How to cite DOPE

If you found our work helpful, consider citing us with the following BibTeX reference:

```
@article{jeffrey2025deeprob,
  title = {DOPE-Plus: Enhancements in Feature Extraction and Data Generation for 6D Pose Estimation},
  author = {Chen, Jeffrey and Luo, Yuqiao and Yuan, Longzhen},
  year = {2025}
}
```

Please cite the original DOPE as well:
```
@inproceedings{tremblay2018corl:dope,
 author = {Jonathan Tremblay and Thang To and Balakumar Sundaralingam and Yu Xiang and Dieter Fox and Stan Birchfield},
 title = {Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects},
 booktitle = {Conference on Robot Learning (CoRL)},
 url = "https://arxiv.org/abs/1809.10790",
 year = 2018
}
```

## Contacts

Jeffrey Chen (jeffzc@umich.edu), Yuqiao Luo (joeluo@umich.edu), and Longzhen Yuan (longzhen@umich.edu)
