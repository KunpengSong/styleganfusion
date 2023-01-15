# styleganfusion

This is the official repository of "Diffusion Guided Domain Adaptation of Image Generators".

## checkpoints to start with:
Please download [stylegan2](https://github.com/rinongal/StyleGAN-nada) and [EG3D](https://github.com/NVlabs/eg3d) checkpoints from [our shared Google Drive](https://drive.google.com/drive/folders/1kY9wEK7hQaO_MGMkeHVmBKfXcfUoYrH4?usp=sharing), and put them in the download folder. They are publicly available checkpoints and are the same as the official ones. 

**_The code is ready to run now after downloading checkpoints._**

A few notices:
+ Please use .sh files as entrances. A requirements.txt is provided to install packages.
+ Results are saved in the logs folder. We also uploaded a lot of log images [here](https://drive.google.com/drive/folders/1l4e7zAu5FwB4wrnUy-EDJC5-sSEZjs_C?usp=share_link) 
+ We use diffusers to load stablediffusion checkpoint from huggingface. A TOKEN file is included to get access to the checkpoints. Please replace it with your own token file, which can be applied [here](https://huggingface.co/CompVis/stable-diffusion-v1-4)

Cheers!

If you find this work useful, please cite:
```
@article{song2022diffusion,
      title={Diffusion Guided Domain Adaptation of Image Generators},
      author={Song, Kunpeng and Han, Ligong and Liu, Bingchen and Metaxas, Dimitris and Elgammal, Ahmed},
      journal={arXiv preprint https://arxiv.org/abs/2212.04473},
      year={2022}
}
```

Acknowledgement: This code is built on [Stylegan-NADA](https://github.com/rinongal/StyleGAN-nada) and [DreamFusion-StableDiffusion](https://github.com/ashawkey/stable-dreamfusion)
