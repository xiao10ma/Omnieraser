# OmniEraser

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://pris-cv.github.io/Omnieraser/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.07397-b31b1b.svg)](https://arxiv.org/pdf/2501.07397)

Code release for "OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data" 


**Abstract**: Inpainting algorithms have achieved remarkable progress in removing objects from images, yet still face two challenges: 1) struggle to handle the object's visual effects such as shadow and reflection; 2) easily generate shape-like artifacts and unintended content.In this paper, we propose `Video4Removal`, a large-scale dataset comprising over 100,000 high-quality samples with realistic object shadows and reflections. By constructing object-background pairs from video frames with off-the-shelf vision models,the labor costs of data acquisition can be significantly reduced.To avoid generating shape-like artifacts and unintended content, we propose Object-Background Guidance, an elaborated paradigm that takes both the foreground object and background images.It can guide the diffusion process to harness richer contextual information.Based on the above two designs, we present `OmniEraser`, a novel method that seamlessly removes objects and their visual effects using only object masks as input. Extensive experiments show that OmniEraser significantly outperforms previous methods, particularly in complex in-the-wild scenes. And it also exhibits a strong generalization ability in anime-style images. 


## TODO

|  | Task                         | Update  |
|----|------------------------------|------------------|
| ✅ | 🖥️ **Training & inference code (Base version)** | 04/01/2025 |
| ✅ | 🖥️ **Training & inference code (ControlNet version)** | 04/01/2025 |
| ☐  | 🤗 **Gradio demo**           | Expected about 10 days |
| ☐  | ⚖️ **Model weights**        | Under active preparation |
| ☐  | 📂 **Dataset**               | Under active preparation |


## Citation
If you find this paper useful in your research, please consider citing:
```
@misc{wei2025omnieraserremoveobjectseffects,
      title={OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data}, 
      author={Runpu Wei and Zijin Yin and Shuo Zhang and Lanxiang Zhou and Xueyi Wang and Chao Ban and Tianwei Cao and Hao Sun and Zhongjiang He and Kongming Liang and Zhanyu Ma},
      year={2025},
      eprint={2501.07397},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.07397}, 
}
```
