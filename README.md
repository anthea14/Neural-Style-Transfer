# Neural Style Transfer with TensorFlow

This notebook implements neural style transfer using TensorFlow and a pre-trained VGG19 model. The algorithm combines the content of one image with the artistic style of another image.

## Overview

Neural style transfer is an optimization technique that takes two images—a content image and a style reference image—and blends them together so the output image looks like the content image but "painted" in the style of the style image.

## How It Works

The implementation uses:
- **VGG19** pre-trained on ImageNet as the feature extractor
- **Gram matrices** to capture style information from style layers
- **Content loss** to preserve the content structure
- **Style loss** to transfer artistic patterns
- **Adam optimizer** to iteratively update the generated image

## Features

- Load and preprocess images with resizing to 512px max dimension
- Extract content features from `block5_conv2` layer
- Extract style features from multiple convolutional layers:
  - block1_conv1
  - block2_conv1
  - block3_conv1
  - block4_conv1
  - block5_conv1
- Configurable training epochs and steps
- Visual comparison of content, style, and result images

## Requirements
```
tensorflow
numpy
matplotlib
Pillow
```
## Usage

1. **Prepare your images**: Place your content image and style image in the notebook's working directory
2. **Update file paths**: Modify the `content_path` and `style_path` variables to point to your images
3. **Run the notebook**: Execute all cells sequentially
4. **View results**: The final cell displays side-by-side comparison of content, style, and styled output

### Customization Parameters

You can adjust the following parameters in the `train_style_transfer()` function:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 10 | Number of training epochs |
| `steps_per_epoch` | 100 | Optimization steps per epoch |
| `style_weight` | 1e-2 | Weight for style loss (higher = more style) |
| `content_weight` | 1e4 | Weight for content loss (higher = more content) |
| `learning_rate` | 0.02 | Adam optimizer learning rate |
| `max_dim` | 512 | Maximum image dimension for resizing |

## Example

```python
content_path = "/content/content_image.jpg"
style_path = "/content/style_image.jpg"
result = train_style_transfer(content_path, style_path, epochs=5, steps_per_epoch=50)
```
## Code Structure
- load_img() - Loads and preprocesses images

- get_vgg_layers() - Creates VGG19 model with specified layer outputs

- StyleContentModel - Custom model for extracting style and content features with Gram matrix computation

- train_style_transfer() - Main training loop implementing the optimization
