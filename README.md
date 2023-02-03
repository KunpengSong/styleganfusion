# styleganfusion

This is the official repository of "[Diffusion Guided Domain Adaptation of Image Generators](https://arxiv.org/abs/2212.04473)".

[Project Page](https://styleganfusion.github.io/)

## checkpoints to start with:
Please download [StyleGAN2](https://github.com/rinongal/StyleGAN-nada) or [EG3D](https://github.com/NVlabs/eg3d) checkpoints from [our shared Google Drive](https://drive.google.com/drive/folders/1kY9wEK7hQaO_MGMkeHVmBKfXcfUoYrH4?usp=sharing), and put them in the download folder. They are publicly available checkpoints and are the same as the official ones. 

**_The code is ready to run now after downloading checkpoints._**

A Few Notes:
+ Please use .sh files as entrances. A requirements.txt is provided to install packages.
+ Results are saved in the logs folder. We also uploaded a lot of log images [here](https://drive.google.com/drive/folders/1l4e7zAu5FwB4wrnUy-EDJC5-sSEZjs_C?usp=share_link).
+ We shared the manually labeled "[AFHQ](https://www.kaggle.com/datasets/andrewmvd/animal-faces)-wild" images by their detailed types(fox/tiger/lion/wolf). See [AFHQ-wild_detailed_labels](https://drive.google.com/drive/folders/1eYx2p5OAhQfcLHiJvmgn1KVXVRSQZVup?usp=share_link). We used it to calculate FIDs. 
+ We use [diffusers](https://huggingface.co/docs/diffusers/installation) to load StableDiffusion V1.4 checkpoint from Huggingface. A TOKEN file is included to get access to the checkpoints. Please replace it with your own token file, which can be applied [here](https://huggingface.co/CompVis/stable-diffusion-v1-4).
+ To calculate FID scores, please use metrics/IS_FID_prdc/src/run_FID.sh as entrance. (credit to [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN))


If you find this work useful, please cite:
```
@article{song2022diffusion,
      title={Diffusion Guided Domain Adaptation of Image Generators},
      author={Song, Kunpeng and Han, Ligong and Liu, Bingchen and Metaxas, Dimitris and Elgammal, Ahmed},
      journal={arXiv preprint https://arxiv.org/abs/2212.04473},
      year={2022}
}
```
Cheers!

Acknowledgement: This code is built on [Stylegan-NADA](https://github.com/rinongal/StyleGAN-nada) and [DreamFusion-StableDiffusion](https://github.com/ashawkey/stable-dreamfusion)
