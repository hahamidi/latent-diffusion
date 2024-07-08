# Stable Diffusion
*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
<<<<<<< HEAD
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1) is a latent text-to-image diffusion
model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), we were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. 
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487), 
this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See [this section](#stable-diffusion-v1) below and the [model card](https://huggingface.co/CompVis/stable-diffusion).

=======
\* equal contribution

<p align="center">
<img src=assets/modelfigure.png />
</p>

## News

### July 2022
- Inference code and model weights to run our [retrieval-augmented diffusion models](https://arxiv.org/abs/2204.11824) are now available. See [this section](#retrieval-augmented-diffusion-models).
### April 2022
- Thanks to [Katherine Crowson](https://github.com/crowsonkb), classifier-free guidance received a ~2x speedup and the [PLMS sampler](https://arxiv.org/abs/2202.09778) is available. See also [this PR](https://github.com/CompVis/latent-diffusion/pull/51).

- Our 1.45B [latent diffusion LAION model](#text-to-image) was integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/multimodalart/latentdiffusion)

- More pre-trained LDMs are available: 
  - A 1.45B [model](#text-to-image) trained on the [LAION-400M](https://arxiv.org/abs/2111.02114) database.
  - A class-conditional model on ImageNet, achieving a FID of 3.6 when using [classifier-free guidance](https://openreview.net/pdf?id=qw8AKxfYbI) Available via a [colab notebook](https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin].
>>>>>>> origin/main
  
## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

<<<<<<< HEAD
You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
``` 


## Stable Diffusion v1

Stable Diffusion v1 refers to a specific configuration of the model
architecture that uses a downsampling-factor 8 autoencoder with an 860M UNet
and CLIP ViT-L/14 text encoder for the diffusion model. The model was pretrained on 256x256 images and 
then finetuned on 512x512 images.

*Note: Stable Diffusion v1 is a general text-to-image diffusion model and therefore mirrors biases and (mis-)conceptions that are present
in its training data. 
Details on the training procedure and data, as well as the intended use of the model can be found in the corresponding [model card](Stable_Diffusion_v1_Model_Card.md).*

The weights are available via [the CompVis organization at Hugging Face](https://huggingface.co/CompVis) under [a license which contains specific use-based restrictions to prevent misuse and harm as informed by the model card, but otherwise remains permissive](LICENSE). While commercial use is permitted under the terms of the license, **we do not recommend using the provided weights for services or products without additional safety mechanisms and considerations**, since there are [known limitations and biases](Stable_Diffusion_v1_Model_Card.md#limitations-and-bias) of the weights, and research on safe and ethical deployment of general text-to-image models is an ongoing effort. **The weights are research artifacts and should be treated as such.**

[The CreativeML OpenRAIL M license](LICENSE) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.

### Weights

We currently provide the following checkpoints:

- `sd-v1-1.ckpt`: 237k steps at resolution `256x256` on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en).
  194k steps at resolution `512x512` on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) (170M examples from LAION-5B with resolution `>= 1024x1024`).
- `sd-v1-2.ckpt`: Resumed from `sd-v1-1.ckpt`.
  515k steps at resolution `512x512` on [laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/) (a subset of laion2B-en with estimated aesthetics score `> 5.0`, and additionally
filtered to images with an original size `>= 512x512`, and an estimated watermark probability `< 0.5`. The watermark estimate is from the [LAION-5B](https://laion.ai/blog/laion-5b/) metadata, the aesthetics score is estimated using the [LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)).
- `sd-v1-3.ckpt`: Resumed from `sd-v1-2.ckpt`. 195k steps at resolution `512x512` on "laion-aesthetics v2 5+" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).
- `sd-v1-4.ckpt`: Resumed from `sd-v1-2.ckpt`. 225k steps at resolution `512x512` on "laion-aesthetics v2 5+" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).

Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,
5.0, 6.0, 7.0, 8.0) and 50 PLMS sampling
steps show the relative improvements of the checkpoints:
![sd evaluation results](assets/v1-variants-scores.jpg)



### Text-to-Image with Stable Diffusion
![txt2img-stable2](assets/stable-samples/txt2img/merged-0005.png)
![txt2img-stable2](assets/stable-samples/txt2img/merged-0007.png)

Stable Diffusion is a latent diffusion model conditioned on the (non-pooled) text embeddings of a CLIP ViT-L/14 text encoder.
We provide a [reference script for sampling](#reference-sampling-script), but
there also exists a [diffusers integration](#diffusers-integration), which we
expect to see more active community development.

#### Reference Sampling Script

We provide a reference sampling script, which incorporates

- a [Safety Checker Module](https://github.com/CompVis/stable-diffusion/pull/36),
  to reduce the probability of explicit outputs,
- an [invisible watermarking](https://github.com/ShieldMnt/invisible-watermark)
  of the outputs, to help viewers [identify the images as machine-generated](scripts/tests/test_watermark.py).
=======
# Pretrained Models
A general list of all available checkpoints is available in via our [model zoo](#model-zoo).
If you use any of these models in your work, we are always happy to receive a [citation](#bibtex).

## Retrieval Augmented Diffusion Models
![rdm-figure](assets/rdm-preview.jpg)
We include inference code to run our retrieval-augmented diffusion models (RDMs) as described in [https://arxiv.org/abs/2204.11824](https://arxiv.org/abs/2204.11824).


To get started, install the additionally required python packages into your `ldm` environment
```shell script
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```
and download the trained weights (preliminary ceckpoints):

```bash
mkdir -p models/rdm/rdm768x768/
wget -O models/rdm/rdm768x768/model.ckpt https://ommer-lab.com/files/rdm/model.ckpt
```
As these models are conditioned on a set of CLIP image embeddings, our RDMs support different inference modes, 
which are described in the following.
#### RDM with text-prompt only (no explicit retrieval needed)
Since CLIP offers a shared image/text feature space, and RDMs learn to cover a neighborhood of a given
example during training, we can directly take a CLIP text embedding of a given prompt and condition on it.
Run this mode via
```
python scripts/knn2img.py  --prompt "a happy bear reading a newspaper, oil on canvas"
```

#### RDM with text-to-image retrieval

To be able to run a RDM conditioned on a text-prompt and additionally images retrieved from this prompt, you will also need to download the corresponding retrieval database. 
We provide two distinct databases extracted from the [Openimages-](https://storage.googleapis.com/openimages/web/index.html) and [ArtBench-](https://github.com/liaopeiyuan/artbench) datasets. 
Interchanging the databases results in different capabilities of the model as visualized below, although the learned weights are the same in both cases. 

Download the retrieval-databases which contain the retrieval-datasets ([Openimages](https://storage.googleapis.com/openimages/web/index.html) (~11GB) and [ArtBench](https://github.com/liaopeiyuan/artbench) (~82MB)) compressed into CLIP image embeddings:
```bash
mkdir -p data/rdm/retrieval_databases
wget -O data/rdm/retrieval_databases/artbench.zip https://ommer-lab.com/files/rdm/artbench_databases.zip
wget -O data/rdm/retrieval_databases/openimages.zip https://ommer-lab.com/files/rdm/openimages_database.zip
unzip data/rdm/retrieval_databases/artbench.zip -d data/rdm/retrieval_databases/
unzip data/rdm/retrieval_databases/openimages.zip -d data/rdm/retrieval_databases/
```
We also provide trained [ScaNN](https://github.com/google-research/google-research/tree/master/scann) search indices for ArtBench. Download and extract via
```bash
mkdir -p data/rdm/searchers
wget -O data/rdm/searchers/artbench.zip https://ommer-lab.com/files/rdm/artbench_searchers.zip
unzip data/rdm/searchers/artbench.zip -d data/rdm/searchers
```

Since the index for OpenImages is large (~21 GB), we provide a script to create and save it for usage during sampling. Note however,
that sampling with the OpenImages database will not be possible without this index. Run the script via
```bash
python scripts/train_searcher.py
```

Retrieval based text-guided sampling with visual nearest neighbors can be started via 
```
python scripts/knn2img.py  --prompt "a happy pineapple" --use_neighbors --knn <number_of_neighbors> 
```
Note that the maximum supported number of neighbors is 20. 
The database can be changed via the cmd parameter ``--database`` which can be `[openimages, artbench-art_nouveau, artbench-baroque, artbench-expressionism, artbench-impressionism, artbench-post_impressionism, artbench-realism, artbench-renaissance, artbench-romanticism, artbench-surrealism, artbench-ukiyo_e]`.
For using `--database openimages`, the above script (`scripts/train_searcher.py`) must be executed before.
Due to their relatively small size, the artbench datasetbases are best suited for creating more abstract concepts and do not work well for detailed text control. 


#### Coming Soon
- better models
- more resolutions
- image-to-image retrieval

## Text-to-Image
![text2img-figure](assets/txt2img-preview.png) 


Download the pre-trained weights (5.7GB)
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```
and sample with
```
python scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```
This will save each sample individually as well as a grid of size `n_iter` x `n_samples` at the specified output location (default: `outputs/txt2img-samples`).
Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` arguments.
As a rule of thumb, higher values of `scale` produce better samples at the cost of a reduced output diversity.   
Furthermore, increasing `ddim_steps` generally also gives higher quality samples, but returns are diminishing for values > 250.
Fast sampling (i.e. low values of `ddim_steps`) while retaining good quality can be achieved by using `--ddim_eta 0.0`.  
Faster sampling (i.e. even lower values of `ddim_steps`) while retaining good quality can be achieved by using `--ddim_eta 0.0` and `--plms` (see [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778)).

#### Beyond 256²

For certain inputs, simply running the model in a convolutional fashion on larger features than it was trained on
can sometimes result in interesting results. To try it out, tune the `H` and `W` arguments (which will be integer-divided
by 8 in order to calculate the corresponding latent size), e.g. run

```
python scripts/txt2img.py --prompt "a sunset behind a mountain range, vector image" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0  
```
to create a sample of size 384x1024. Note, however, that controllability is reduced compared to the 256x256 setting. 

The example below was generated using the above command. 
![text2img-figure-conv](assets/txt2img-convsample.png)



## Inpainting
![inpainting](assets/inpainting.png)
>>>>>>> origin/main

After [obtaining the `stable-diffusion-v1-*-original` weights](#weights), link them
```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```
and sample with
```
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
```
<<<<<<< HEAD
=======
`indir` should contain images `*.png` and masks `<image_fname>_mask.png` like
the examples provided in `data/inpainting_examples`.

## Class-Conditional ImageNet

Available via a [notebook](scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin].
![class-conditional](assets/birdhouse.png)

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-cin]: <https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb>


## Unconditional Models

We also provide a script for sampling from unconditional LDMs (e.g. LSUN, FFHQ, ...). Start it via

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta> 
```

# Train your own LDMs

## Data preparation

### Faces 
For downloading the CelebA-HQ and FFHQ datasets, proceed as described in the [taming-transformers](https://github.com/CompVis/taming-transformers#celeba-hq) 
repository.

### LSUN 

The LSUN datasets can be conveniently downloaded via the script available [here](https://github.com/fyu/lsun).
We performed a custom split into training and validation images, and provide the corresponding filenames
at [https://ommer-lab.com/files/lsun.zip](https://ommer-lab.com/files/lsun.zip). 
After downloading, extract them to `./data/lsun`. The beds/cats/churches subsets should
also be placed/symlinked at `./data/lsun/bedrooms`/`./data/lsun/cats`/`./data/lsun/churches`, respectively.

### ImageNet
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:
>>>>>>> origin/main

By default, this uses a guidance scale of `--scale 7.5`, [Katherine Crowson's implementation](https://github.com/CompVis/latent-diffusion/pull/51) of the [PLMS](https://arxiv.org/abs/2202.09778) sampler, 
and renders images of size 512x512 (which it was trained on) in 50 steps. All supported arguments are listed below (type `python scripts/txt2img.py --help`).


```commandline
usage: txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--skip_grid] [--skip_save] [--ddim_steps DDIM_STEPS] [--plms] [--laion400m] [--fixed_code] [--ddim_eta DDIM_ETA]
                  [--n_iter N_ITER] [--H H] [--W W] [--C C] [--f F] [--n_samples N_SAMPLES] [--n_rows N_ROWS] [--scale SCALE] [--from-file FROM_FILE] [--config CONFIG] [--ckpt CKPT]
                  [--seed SEED] [--precision {full,autocast}]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     the prompt to render
  --outdir [OUTDIR]     dir to write results to
  --skip_grid           do not save a grid, only individual samples. Helpful when evaluating lots of samples
  --skip_save           do not save individual samples. For speed measurements.
  --ddim_steps DDIM_STEPS
                        number of ddim sampling steps
  --plms                use plms sampling
  --laion400m           uses the LAION400M model
  --fixed_code          if enabled, uses the same starting code across samples
  --ddim_eta DDIM_ETA   ddim eta (eta=0.0 corresponds to deterministic sampling
  --n_iter N_ITER       sample this often
  --H H                 image height, in pixel space
  --W W                 image width, in pixel space
  --C C                 latent channels
  --f F                 downsampling factor
  --n_samples N_SAMPLES
                        how many samples to produce for each given prompt. A.k.a. batch size
  --n_rows N_ROWS       rows in the grid (default: n_samples)
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --from-file FROM_FILE
                        if specified, load prompts from this file
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of model
  --seed SEED           the seed (for reproducible sampling)
  --precision {full,autocast}
                        evaluate at this precision
```
Note: The inference config for all v1 versions is designed to be used with EMA-only checkpoints. 
For this reason `use_ema=False` is set in the configuration, otherwise the code will try to switch from
non-EMA to EMA weights. If you want to examine the effect of EMA vs no EMA, we provide "full" checkpoints
which contain both types of weights. For these, `use_ema=False` will load and use the non-EMA weights.


#### Diffusers Integration

A simple way to download and sample Stable Diffusion is by using the [diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers):
```py
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  
    
image.save("astronaut_rides_horse.png")
```


### Image Modification with Stable Diffusion

By using a diffusion-denoising mechanism as first proposed by [SDEdit](https://arxiv.org/abs/2108.01073), the model can be used for different 
tasks such as text-guided image-to-image translation and upscaling. Similar to the txt2img sampling script, 
we provide a script to perform image modification with Stable Diffusion.  

The following describes an example where a rough sketch made in [Pinta](https://www.pinta-project.com/) is converted into a detailed artwork.
```
python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8
```
Here, strength is a value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input. See the following example.

**Input**

![sketch-in](assets/stable-samples/img2img/sketch-mountains-input.jpg)

**Outputs**

<<<<<<< HEAD
![out3](assets/stable-samples/img2img/mountains-3.png)
![out2](assets/stable-samples/img2img/mountains-2.png)

This procedure can, for example, also be used to upscale samples from the base model.

=======
# Model Zoo 

## Pretrained Autoencoding Models
![rec2](assets/reconstruction2.png)

All models were trained until convergence (no further substantial improvement in rFID).

| Model                   | rFID vs val | train steps           |PSNR           | PSIM          | Link                                                                                                                                                  | Comments              
|-------------------------|------------|----------------|----------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| f=4, VQ (Z=8192, d=3)   | 0.58       | 533066 | 27.43  +/- 4.26 | 0.53 +/- 0.21 |     https://ommer-lab.com/files/latent-diffusion/vq-f4.zip                   |  |
| f=4, VQ (Z=8192, d=3)   | 1.06       | 658131 | 25.21 +/-  4.17 | 0.72 +/- 0.26 | https://heibox.uni-heidelberg.de/f/9c6681f64bb94338a069/?dl=1  | no attention          |
| f=8, VQ (Z=16384, d=4)  | 1.14       | 971043 | 23.07 +/- 3.99 | 1.17 +/- 0.36 |       https://ommer-lab.com/files/latent-diffusion/vq-f8.zip                     |                       |
| f=8, VQ (Z=256, d=4)    | 1.49       | 1608649 | 22.35 +/- 3.81 | 1.26 +/- 0.37 |   https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip |  
| f=16, VQ (Z=16384, d=8) | 5.15       | 1101166 | 20.83 +/- 3.61 | 1.73 +/- 0.43 |             https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1                        |                       |
|                         |            |  |                |               |                                                                                                                                                    |                       |
| f=4, KL                 | 0.27       | 176991 | 27.53 +/- 4.54 | 0.55 +/- 0.24 |     https://ommer-lab.com/files/latent-diffusion/kl-f4.zip                                   |                       |
| f=8, KL                 | 0.90       | 246803 | 24.19 +/- 4.19 | 1.02 +/- 0.35 |             https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                            |                       |
| f=16, KL     (d=16)     | 0.87       | 442998 | 24.08 +/- 4.22 | 1.07 +/- 0.36 |      https://ommer-lab.com/files/latent-diffusion/kl-f16.zip                                  |                       |
 | f=32, KL     (d=64)     | 2.04       | 406763 | 22.27 +/- 3.93 | 1.41 +/- 0.40 |             https://ommer-lab.com/files/latent-diffusion/kl-f32.zip                            |                       |

### Get the models

Running the following script downloads und extracts all available pretrained autoencoding models.   
```shell script
bash scripts/download_first_stages.sh
```

The first stage models can then be found in `models/first_stage_models/<model_spec>`



## Pretrained LDMs
| Datset                          |   Task    | Model        | FID           | IS              | Prec | Recall | Link                                                                                                                                                                                   | Comments                                        
|---------------------------------|------|--------------|---------------|-----------------|------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| CelebA-HQ                       | Unconditional Image Synthesis    |  LDM-VQ-4 (200 DDIM steps, eta=0)| 5.11 (5.11)          | 3.29            | 0.72    | 0.49 |    https://ommer-lab.com/files/latent-diffusion/celeba.zip     |                                                 |  
| FFHQ                            | Unconditional Image Synthesis    |  LDM-VQ-4 (200 DDIM steps, eta=1)| 4.98 (4.98)  | 4.50 (4.50)   | 0.73 | 0.50 |              https://ommer-lab.com/files/latent-diffusion/ffhq.zip                                              |                                                 |
| LSUN-Churches                   | Unconditional Image Synthesis   |  LDM-KL-8 (400 DDIM steps, eta=0)| 4.02 (4.02) | 2.72 | 0.64 | 0.52 |         https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip        |                                                 |  
| LSUN-Bedrooms                   | Unconditional Image Synthesis   |  LDM-VQ-4 (200 DDIM steps, eta=1)| 2.95 (3.0)          | 2.22 (2.23)| 0.66 | 0.48 | https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip |                                                 |  
| ImageNet                        | Class-conditional Image Synthesis | LDM-VQ-8 (200 DDIM steps, eta=1) | 7.77(7.76)* /15.82** | 201.56(209.52)* /78.82** | 0.84* / 0.65** | 0.35* / 0.63** |   https://ommer-lab.com/files/latent-diffusion/cin.zip                                                                   | *: w/ guiding, classifier_scale 10  **: w/o guiding, scores in bracket calculated with script provided by [ADM](https://github.com/openai/guided-diffusion) |   
| Conceptual Captions             |  Text-conditional Image Synthesis | LDM-VQ-f4 (100 DDIM steps, eta=0) | 16.79         | 13.89           | N/A | N/A |              https://ommer-lab.com/files/latent-diffusion/text2img.zip                                | finetuned from LAION                            |   
| OpenImages                      | Super-resolution   | LDM-VQ-4     | N/A            | N/A               | N/A    | N/A    |                                    https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip                                    | BSR image degradation                           |
| OpenImages                      | Layout-to-Image Synthesis    | LDM-VQ-4 (200 DDIM steps, eta=0) | 32.02         | 15.92           | N/A    | N/A    |                  https://ommer-lab.com/files/latent-diffusion/layout2img_model.zip                                           |                                                 | 
| Landscapes      |  Semantic Image Synthesis   | LDM-VQ-4  | N/A             | N/A               | N/A    | N/A    |           https://ommer-lab.com/files/latent-diffusion/semantic_synthesis256.zip                                    |                                                 |
| Landscapes       |  Semantic Image Synthesis   | LDM-VQ-4  | N/A             | N/A               | N/A    | N/A    |           https://ommer-lab.com/files/latent-diffusion/semantic_synthesis.zip                                    |             finetuned on resolution 512x512                                     |


### Get the models

The LDMs listed above can jointly be downloaded and extracted via

```shell script
bash scripts/download_models.sh
```

The models can then be found in `models/ldm/<model_spec>`.



## Coming Soon...

* More inference scripts for conditional LDMs.
* In the meantime, you can play with our colab notebook https://colab.research.google.com/drive/1xqzUi2iXQXDqXBHQGP9Mqt2YrYW6cx-J?usp=sharing
>>>>>>> origin/main

## Comments 

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}


```


