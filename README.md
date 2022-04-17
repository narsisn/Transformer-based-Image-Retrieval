# Transformer-based-Image-Retrieval
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. In this repository I have used the pretained CLIP models (https://huggingface.co/openai/clip-vit-large-patch14) for visual search downstream task. 

# Usage 

**Requirement Packages**
```
python3.*
import sys
sys.path.append(str(Path('.').absolute().parent))

from pathlib import Path
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer,CLIPConfig
from src import SNAPDemo
import glob

```
