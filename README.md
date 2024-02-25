# DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis
### [Project Page](https://vcai.mpi-inf.mpg.de/projects/DELIFFAS/) | [Video](https://www.youtube.com/watch?v=OpIMnSS26pc) | [Paper](https://arxiv.org/pdf/2310.11449.pdf)


<img src="https://youngjoongunc.github.io/papers/images/deliffas_teaser.gif" width="70%" height="70%" />

> [DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis (NeurIPS 2023)](https://arxiv.org/pdf/2012.15838.pdf)  
> Youngjoong Kwon, Lingjie Liu, Henry Fuchs, Marc Habermann✝, Christian Theobalt  
> ✝Corresponding author.

## Overview

This codebase contains the official implementation of DELIFFAS and the VCAI/GVDH differentiable rasterizer and custom TF operators (from DeepCap and DDC projects).



## Installation
This codebase requires the installation of the differentiable rasterizer and custom TF operators.
Please see [INSTALL.md](INSTALL.md).

Please download the Vlad, FranziRed, and Olek datasets from [here](https://gvv-assets.mpi-inf.mpg.de/ddc/?page_id=11&redirect_to=https%3A%2F%2Fgvv-assets.mpi-inf.mpg.de%2Fddc%2F).

## Training

1. Train model on 'Vlad' subject
    ```
    CUDA_VISIBLE_DEVICES=0 python mainTestTrain.py --config Configs/model_main/config_vlad_train.sh
    ```

## Evaluation

1. Test (quantitative eval, free-view rendering) model on 'Vlad' subject. Note that quantitative evaluation will be done to the every 10-th frame.
    ```
    CUDA_VISIBLE_DEVICES=0 python mainTestTrain.py --config Configs/model_main/config_vlad_test.sh
    ```
2. Compute the evaluation metrics
    ```
    cd Projects/EvaluationScripts   
    CUDA_VISIBLE_DEVICES=0 python Lpips.py
    ```
## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{kwon2023deliffas,
title = {DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis},
author = {Kwon, Youngjoong and Liu, Lingjie and Fuchs, Henry and Habermann, Marc and Theobalt, Christian},
year = {2023},
journal={Advances in Neural Information Processing Systems}
}

@article{habermann2021real,
  title={Real-time deep dynamic characters},
  author={Habermann, Marc and Liu, Lingjie and Xu, Weipeng and Zollhoefer, Michael and Pons-Moll, Gerard and Theobalt, Christian},
  journal={ACM Transactions on Graphics (ToG)},
  volume={40},
  number={4},
  pages={1--16},
  year={2021},
  publisher={ACM New York, NY, USA}
}

@inproceedings{habermann2020deepcap,
  title={Deepcap: Monocular human performance capture using weak supervision},
  author={Habermann, Marc and Xu, Weipeng and Zollhofer, Michael and Pons-Moll, Gerard and Theobalt, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5052--5063},
  year={2020}
}
```

## Acknowledgments
Christian Theobalt and Marc Habermann were supported by ERC Consolidator Grant 4DReply (770784). Lingjie Liu was supported by Lise Meitner Postdoctoral Fellowship. This work was partially supported by National Science Foundation Award 2107454.

## Contact

For questions, please contact [youngjo2@andrew.cmu.edu](youngjo2@andrew.cmu.edu).
