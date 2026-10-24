# Image Enhancement, Generation and Style Transfer

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/bodyastar/image-enhancement-generation-style-transfer)

ESRGAN super-resolution, Stable Diffusion v1.5 and VGG19 neural style transfer, compared in one reproducible notebook.

## Overview

The notebook `image_enhancement_generation_style_transfer.ipynb` explores three families of models for images:

| Part | Task | Model | Library |
|---|---|---|---|
| 1 | 4x super-resolution | ESRGAN (`esrgan-tf2`) | TensorFlow Hub |
| 2 | Text-to-image and image-to-image generation | Stable Diffusion v1.5 | Hugging Face `diffusers`, PyTorch |
| 3 | Neural style transfer | VGG19 (ImageNet weights) | TensorFlow / Keras |

## What is inside

- **ESRGAN:** side-by-side comparison with the originals, and a quantitative benchmark (PSNR and SSIM against bicubic interpolation) on simulated 4x downscaled images.
- **Stable Diffusion:** six styled prompts, and controlled experiments that change one parameter at a time with a fixed seed: guidance scale, negative prompt and the img2img `strength`.
- **Style transfer:** Gram-matrix style loss, content loss and total variation regularization, with a recorded loss curve.

## Results

### ESRGAN vs. bicubic interpolation

Each test image was downscaled 4x with bicubic interpolation, then upscaled back with bicubic interpolation and with ESRGAN, and compared with the original (15 images).

| Metric | Bicubic | ESRGAN |
|---|---|---|
| Mean PSNR (dB) | 24.49 | **25.33** |
| Mean SSIM | 0.6956 | **0.7359** |

ESRGAN achieved a higher SSIM on all 15 images and a higher PSNR on 14 of 15. The gain varied by image, from almost zero to about +2.8 dB PSNR. Per-image values are printed in the notebook.

Caveats: the test degradation (bicubic downscaling) is the one the model was trained for, so the numbers say little about real-world artifacts such as JPEG compression or noise. The input files are themselves low-resolution (x4) versions, so the references are not true high-resolution images, and the absolute values cannot be compared with published benchmarks.

### Stable Diffusion

Text-to-image samples (512x512): realistic portrait, anime, fantasy, cyberpunk, nature and digital art. The model still makes visible mistakes, for example the duplicated face in the portrait.

![Stable Diffusion examples](images/stable_diffusion_examples.jpg)

The notebook also compares guidance scales (3, 7, 12, 20), generation with and without a negative prompt, and img2img strengths (0.3, 0.5, 0.75, 0.95), all with a fixed seed. These results are evaluated qualitatively.

### Neural style transfer

500 optimization steps (10 epochs of 50 steps) took 37.6 s on a Kaggle GPU; the final total loss was about 3.73 x 10^6. The loss combines three weighted terms, so its absolute value is not meaningful on its own and serves mainly as a convergence check.

## Setup

**Option 1: Run on Kaggle (recommended).** Open the [Kaggle notebook](https://www.kaggle.com/code/bodyastar/image-enhancement-generation-style-transfer), enable a **GPU** accelerator and turn **Internet** on (models are downloaded from TensorFlow Hub and the Hugging Face Hub). Attach the input data described in the next section.

**Option 2: Run locally.**

1. Install the dependencies:

   ```bash
   pip install tensorflow tensorflow-hub keras torch diffusers transformers accelerate matplotlib pandas pillow
   ```

2. Download the input images and edit the paths at the top of the notebook (`ESRGAN_INPUT_DIR`, `IMG2IMG_INPUT`, `STYLE_DATA_DIR`, `OUTPUT_DIR`).
3. A CUDA GPU is required for Stable Diffusion.

Outputs (comparison figures, generated images, the stylized image) are saved to the directory set in `OUTPUT_DIR`.

## Data

The repository does not include any input images.

- **ESRGAN test images:** the first 15 images (`0801x4.png` to `0815x4.png`) of the DIV2K validation set, bicubic x4 low-resolution version. DIV2K is made available for academic research purposes only, so the images are **not** redistributed here. Download the dataset from the [official DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and place the images in the folder set in `ESRGAN_INPUT_DIR`.
- **Image-to-image input:** a single demonstration image (`girl.png`) from the Kaggle dataset [img2imgprediffusedimages](https://www.kaggle.com/datasets/bodyastar/img2imgprediffusedimages). The image is not redistributed in this repository; use any image you have the right to process.
- **Style transfer images:** [images-for-style-transfer](https://www.kaggle.com/datasets/soumikrakshit/images-for-style-transfer) by Soumik Rakshit (the content image `TestCases/22714.jpg` and the artwork `Artworks/108700.jpg`). The images are not redistributed here. Check the dataset page for its license and the copyright status of the individual artworks before reusing them.

If you cite the ESRGAN benchmark data, please also cite:

> Agustsson, E., Timofte, R. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.* CVPR Workshops, 2017.
>
> Timofte, R. et al. *NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results.* CVPR Workshops, 2017.

## Models and licenses

| Model | Source | License |
|---|---|---|
| ESRGAN | [Kaggle Models](https://www.kaggle.com/models/kaggle/esrgan-tf2) (formerly TensorFlow Hub, `captain-pool/esrgan-tf2`) | See the license on the model page |
| Stable Diffusion v1.5 | Hugging Face (community mirror of the original weights) | CreativeML OpenRAIL-M, which includes use-based restrictions |
| VGG19 | Keras Applications, ImageNet weights ported from the Oxford Visual Geometry Group | Creative Commons Attribution (CC BY) |

The original `runwayml/stable-diffusion-v1-5` repository was removed from the Hugging Face Hub, so the notebook loads a mirror (`stable-diffusion-v1-5/stable-diffusion-v1-5`). If it becomes unavailable, change `SD_MODEL_ID` in the notebook.

## Limitations

- ESRGAN expects inputs of at least 64x64 px and was trained on bicubic-downscaled images, so it does not necessarily repair real-world degradation such as strong compression artifacts or noise.
- PSNR and SSIM measure pixel fidelity, not perceptual quality, and may disagree with how sharp the result looks. The benchmark uses only 15 images.
- Stable Diffusion results are evaluated qualitatively. The model can produce distorted anatomy, duplicated features and unreadable text.
- Style transfer is an optimization per image: it is slow compared with feed-forward style transfer models and sensitive to the loss weights.
- Generated images can reflect biases of the training data. Do not use photos of real people in image-to-image generation without their consent.

## Reproducibility

Seeds are fixed for Python, NumPy, TensorFlow and PyTorch, and every Stable Diffusion generation uses a seeded generator. Results can still vary with library versions and GPU hardware.

## Future improvements

- Evaluate on a larger set with true high-resolution references and perceptual metrics such as LPIPS.
- Compare ESRGAN with Real-ESRGAN on images with real-world degradation.
- Replace per-image optimization with a fast feed-forward style transfer model.
- Study systematically how the style, content and total variation weights affect the result.

## References

- Goodfellow, I. et al. *Generative Adversarial Networks.* arXiv:1406.2661, 2014.
- Wang, X. et al. *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.* arXiv:1809.00219, 2018.
- Rombach, R. et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* arXiv:2112.10752, 2021.
- Gatys, L. A., Ecker, A. S., Bethge, M. *A Neural Algorithm of Artistic Style.* arXiv:1508.06576, 2015.
- Simonyan, K., Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556, 2014.
- Agustsson, E., Timofte, R. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.* CVPR Workshops, 2017.
- Timofte, R. et al. *NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results.* CVPR Workshops, 2017.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Input data and pretrained models keep their original licenses (see above).
