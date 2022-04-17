# Transformer-based-Image-Retrieval
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. In this repository I have used the pretained CLIP models (https://huggingface.co/openai/clip-vit-large-patch14) for visual search downstream task. 

# Usage 

**Requirement Packages:**
```python
import sys
sys.path.append(str(Path('.').absolute().parent))

from pathlib import Path
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer,CLIPConfig
from src import SNAPDemo
import glob

```

**Downloading pretrained models**

```python
config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
vision_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', config=config.vision_config)
```
**Generate the image embeddings**
```python
imgDirectory = '/path/to/image_directory'
image_path = glob.glob(imgDirectory, recursive=True)
demo = SNAPDemo(vision_encoder)
demo.compute_image_embeddings(image_path)

```
**Run Image Search**
```python
imgPath = 'path/to/search_iamge.jpeg'
demo.image_search(imgPath,10)
```
![image](https://user-images.githubusercontent.com/41056415/163726743-bdcdb191-9c11-4258-8a3d-93e51d81ace3.png)

![image](https://user-images.githubusercontent.com/41056415/163727541-a298ee65-29dd-4d81-89c1-62f017be9309.png)

![image](https://user-images.githubusercontent.com/41056415/163727595-de74f6f5-9cdd-45b8-9a03-e0a0c0b18dc2.png)

![image](https://user-images.githubusercontent.com/41056415/163727720-ca493aa8-cbe5-4adc-8e96-eb20877a0be3.png)
