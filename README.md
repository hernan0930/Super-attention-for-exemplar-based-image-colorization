# Super-attention-for-exemplar-based-image-colorization

**Super-attention for exemplar-based image colorization** <br>
*Hernan Carrillo, Michaël Clément, Aurélie Bugeau.* <br>
Asian Conference on Computer Vision (ACCV), 2022 <br>
[[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Camilo_Super-attention_for_exemplar-based_image_colorization_ACCV_2022_paper.pdf)] [[Supplementary](https://hal.science/hal-03794455v1/file/Super_attention_for_exemplar_based_image_colorization_supplementary_materials.pdf)]

## Reference

Citation:

```latex
@inproceedings{carrillo2022super,
  title={Super-attention for exemplar-based image colorization},
  author={Carrillo, Hernan and Cl{\'e}ment, Micha{\"e}l and Bugeau, Aur{\'e}lie},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  pages={4548--4564},
  year={2022}
}
```

## Requirements [TO DO]

- python==3.6.13
- pytorch==1.7.1
- torchvision==0.8.2
- pillow==8.3.1

```
conda create -n pdnla python=3.6
conda activate pdnla
conda install pytorch==1.7.1 torchvision==0.8.2  cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Pretrained Model [TO DO]

We uploaded the [pre-trained model]() to Google drive.

## Test [TO DO]

```python
python generate_image.py --files1 ./test_image/contents --files2 ./test_image/color --ckpt_dir <> --result_dir ./result/
```

## Abstract

In image colorization, exemplar-based methods use a reference color image to guide the colorization of a target grayscale image. In this article, we present a deep learning framework for exemplar-based image colorization which relies on attention layers to capture robust correspondences between high-resolution deep features from pairs of images. To avoid the quadratic scaling problem from classic attention, we rely on a novel attention block computed from superpixel features, which we call super-attention. Super-attention blocks can learn to transfer semantically related color characteristics from a reference image at different scales of a deep network. Our experimental validations highlight the interest of this approach for exemplar-based colorization. We obtain promising results, achieving visually appealing colorization and outperforming state-of-theart methods on different quantitative metrics.

## Our Framework

Diagram of our proposal for exemplar-based image colorization.:

![framework](./diagrams_img/diagram_net-2_0.png)

Diagram of our super-attention block. This layer takes a reference luminance
feature map f_R, reference color feature map φ_R and a target luminance feature map
f_T, as an input, and learns an attention map at superpixel level by means of a robust
matching between high-resolution encoded feature maps:

![super_attent](./diagrams_img/super-attention_unpooling.png)
