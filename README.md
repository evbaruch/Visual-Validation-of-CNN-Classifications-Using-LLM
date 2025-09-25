# Visual Validation of CNN Classifications Using LLM

This project evaluates Explainable AI (XAI) methods for CNN image classification by using Large Language Models (LLMs) to validate whether masked/corrupted images still contain sufficient visual information for correct classification.

## Project Overview

The core methodology involves:
1. **Image Masking**: Apply various XAI methods (GradCAM, Saliency, etc.) to generate attention maps
2. **Progressive Masking**: Mask images at different thresholds (5%-95%) based on attention maps
3. **LLM Validation**: Use vision-language models to determine if masked images are still recognizable
4. **Evaluation**: Compare XAI method effectiveness by measuring how much visual information can be removed while maintaining recognizability

## Repository Structure

```
├── xai_with_config.py          # Main XAI processing pipeline
├── config_*.json               # Configuration files for different datasets
├── xai_cnn_eval.py            # CNN model evaluation on masked images
├── LlmMain.py                 # Main LLM interface and classification
├── LmmApi/
│   ├── LLMInterface.py        # Core LLM interaction logic
│   ├── LLMStrategy.py         # Abstract strategy pattern for LLMs
│   └── llama32Vision11b.py    # Llama Vision model implementation
├── results.py                 # Result analysis and visualization
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For Llama Vision models, install Ollama
# Follow instructions at: https://ollama.ai/
ollama pull llama3.2-vision:11b
```

### 2. Dataset Preparation

Organize your dataset following one of these structures:

**Option A: Folder-based labels** (for custom datasets like Cervical Cancer):
```
data/source/YourDataset/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
└── class2/
    ├── image3.jpg
    └── image4.jpg
```

**Option B: Filename-based labels** (for ImageNet-style datasets):
```
data/source/imagenet_sample/
└── JPEG/
    ├── 001_goldfish.jpg
    ├── 002_shark.jpg
    └── ...
```

### 3. Configuration

Create a configuration file based on your dataset:

```json
{
  "input_folder": "data/source/YourDataset",
  "mid_folder": "data/mid_YourDataset",
  "model_name": "resnet18",
  "weights_path": "path/to/your/model.pth",  // null for pretrained
  "num_classes": 5,
  "thresholds": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
  "label_extraction": {
    "method": "folder"  // or "filename"
  }
}
```

### 4. Pipeline Execution

**Step 1: Generate Masked Images**
```bash
python xai_with_config.py config_your_dataset.json
```

This creates masked versions of your images using different XAI methods at various masking thresholds.

**Step 2: LLM Evaluation**
```python
from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
from pydantic import BaseModel

class ImageDescription_Boolean(BaseModel):
    boolean: bool

llama = llama32Vision11b()
llm_context = LLMInterface(llama)
llm_context.set_background("You are an image classifier...")
llm_context.set_jsonDescription(ImageDescription_Boolean)

# Process all masked images
llm_context.boolean_outputs_classification_reverse2(
    "data/mid_YourDataset", 
    "data/llm_answers_YourDataset"
)
```

**Step 3: Analysis and Visualization**
```python
import results

# Generate success rate plots and summary tables
results.plot_success_rate_and_dump_table_swapped("data/llm_answers_YourDataset")
```

## Key Components

### XAI Methods Implemented

- **GradCAM**: Gradient-weighted Class Activation Mapping
- **Guided Backpropagation**: Modified backpropagation with guided ReLUs
- **Saliency Maps**: Input gradient magnitudes
- **Integrated Gradients**: Path integral of gradients
- **GradientSHAP**: Gradient-based SHAP values
- **Input × Gradient**: Element-wise input-gradient product
- **SmoothGrad**: Noise-averaged gradients
- **Random**: Baseline random masking

### Model Support

- **Pretrained Models**: ImageNet-pretrained ResNet-18
- **Custom Models**: Support for custom trained models with checkpoint loading
- **Transfer Learning**: Automatic handling of modified classification heads

### LLM Integration

The project uses a strategy pattern for LLM integration:
- **Current**: Llama 3.2 Vision (11B) via Ollama
- **Extensible**: Easy to add OpenAI GPT-4V, Claude Vision, etc.
- **Structured Output**: Pydantic models for reliable JSON responses

## Understanding the Results

### Output Structure

```
data/llm_answers_YourDataset/
├── gradcam/
│   ├── gradcam_05.csv  # 5% masking threshold
│   ├── gradcam_10.csv  # 10% masking threshold
│   └── ...
├── saliency/
│   └── ...
└── success_rate_summary.csv  # Aggregated results
```

### Key Metrics

- **Match**: Boolean indicating if LLM correctly identified the image class
- **Success Rate**: Percentage of correct identifications at each masking threshold
- **Robustness**: How much masking an XAI method can handle while maintaining recognizability

### Interpretation

- **Higher success rates** at high masking thresholds indicate better XAI methods
- **Gradual degradation** suggests the XAI method captures important features
- **Sudden drops** may indicate the method focuses on irrelevant details

## Common Issues and Solutions

### Memory Issues
- Reduce batch size in `xai_with_config.py`
- Set `"reduce_samples": true` in config
- Process images in smaller batches

### CUDA Errors
- Ensure PyTorch CUDA version matches your GPU drivers
- Reduce image resolution if needed
- Use CPU mode: `device = torch.device("cpu")`

### LLM Connection Issues
- Verify Ollama is running: `ollama ps`
- Check model is downloaded: `ollama list`
- Restart Ollama service if needed

### Missing Images
- Check file extensions match your dataset (.jpg, .png, .jpeg)
- Verify folder structure matches configuration
- Enable verbose logging to trace processing

## Extending the Project

### Adding New XAI Methods

1. Implement the method in `xai_with_config.py`:
```python
def your_new_method_mask(model, x, cls, ...):
    # Your implementation
    return attribution_map
```

2. Add to the methods dictionary:
```python
methods = {
    "your_method": lambda: your_new_method_mask(model, x, class_idx, ...),
    # ... existing methods
}
```

### Adding New LLM Providers

1. Create a new strategy class:
```python
from LmmApi.LLMStrategy import LLMStrategy

class YourLLMProvider(LLMStrategy):
    def generate_response(self, prompt, background, image, jsonDescription):
        # Your implementation
        return response
```

2. Use in your analysis:
```python
your_llm = YourLLMProvider()
llm_context = LLMInterface(your_llm)
```

### Custom Datasets

1. Update the configuration file with your dataset structure
2. Modify `get_category_from_path()` if needed for custom label extraction
3. Ensure your trained model checkpoint is compatible

## Research Applications

This framework has been used to evaluate:
- **Medical Image Analysis**: Cervical cancer cell classification
- **General Computer Vision**: ImageNet classification robustness
- **XAI Method Comparison**: Quantitative comparison of explanation quality
- **Model Validation**: Understanding what CNNs actually learn

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{visual-validation-xai,
  title={Visual Validation of CNN Classifications Using LLM},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Future Work Ideas

- **Multi-modal evaluation**: Compare visual and textual explanations
- **Interactive exploration**: Web interface for real-time XAI comparison
- **Automated threshold selection**: ML-based optimal masking threshold detection
- **Cross-dataset generalization**: Evaluate XAI methods across different domains
- **Attention mechanism comparison**: Compare CNN attention with transformer attention

---

For questions or issues, please open a GitHub issue or contact the maintainers.