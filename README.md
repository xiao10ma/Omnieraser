# OmniEraser

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://pris-cv.github.io/Omnieraser/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.07397-b31b1b.svg)](https://arxiv.org/pdf/2501.07397)
[![Hugging Face](https://img.shields.io/badge/Demo-🤗%20Hugging%20Face-blue)](https://huggingface.co/spaces/theSure/Omnieraser)

Code release for "OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data" 

**Abstract**: Inpainting algorithms have achieved remarkable progress in removing objects from images, yet still face two challenges: 1) struggle to handle the object's visual effects such as shadow and reflection; 2) easily generate shape-like artifacts and unintended content.In this paper, we propose `Video4Removal`, a large-scale dataset comprising over 100,000 high-quality samples with realistic object shadows and reflections. By constructing object-background pairs from video frames with off-the-shelf vision models,the labor costs of data acquisition can be significantly reduced.To avoid generating shape-like artifacts and unintended content, we propose Object-Background Guidance, an elaborated paradigm that takes both the foreground object and background images.It can guide the diffusion process to harness richer contextual information.Based on the above two designs, we present `OmniEraser`, a novel method that seamlessly removes objects and their visual effects using only object masks as input. Extensive experiments show that OmniEraser significantly outperforms previous methods, particularly in complex in-the-wild scenes. And it also exhibits a strong generalization ability in anime-style images. 


# News
- **04/08/2025**: 🤗 We have release our fantastic model at [Gradio demo](https://huggingface.co/theSure/Omnieraser/tree/main) !!
- **04/06/2025**: ⚖️ [Omnieraser Model weights (Base version) ](https://huggingface.co/theSure/Omnieraser/tree/main) is released!
- **04/02/2025**: 🖥️ [Omnieraser Training & inference code (ControlNet version) ](https://github.com/PRIS-CV/Omnieraser/tree/main/ControlNet_version) is released!
- **04/01/2025**: 🖥️ [Omnieraser Training & inference code (Base version) ](https://github.com/PRIS-CV/DemoFusion) is released!
- **03/15/2025**: 🔥 Our [Project Page](https://pris-cv.github.io/Omnieraser/) has been published!


## TODO List

|  | Task                         | Update  |
|----|------------------------------|------------------|
| ✅ | 🖥️ **Training & inference code (Base version)** | Released at 04/01/2025 |
| ✅ | 📌 **Training & inference code (ControlNet version)** | Released at 04/02/2025 |
| ✅  | 🤗 **Gradio demo**           | Released at 04/08/2025 |
| ✅  | ⚖️ **Model weights**        | Released at 04/06/2025 |
| ☐  | 📂 **Dataset**               | Under active preparation |


## Citation
If you find this paper useful in your research, please consider citing our paper:
```
@article{wei2025omnieraserremoveobjectseffects,
title={OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data},
author={Runpu Wei and Zijin Yin and Shuo Zhang and Lanxiang Zhou and Xueyi Wang and Chao Ban and Tianwei Cao and Hao Sun and Zhongjiang He and Kongming Liang and Zhanyu Ma},
journal={arXiv preprint arXiv:2501.07397},
year={2025},
url={https://arxiv.org/abs/2501.07397},
}
```
