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

