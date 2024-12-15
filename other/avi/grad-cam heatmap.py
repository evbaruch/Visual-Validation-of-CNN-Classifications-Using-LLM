import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from captum.attr import LayerGradCam

# Load the pretrained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()

# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Load the input image
image_path = r'c:\cats_and_dogs_filtered\train\cats\cat.15.jpg'
img = Image.open(image_path)
transformed_img = transform(img)
input_tensor = transform_normalize(transformed_img).unsqueeze(0)

# Get the model prediction
output = model(input_tensor)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = 'cat'
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

# Apply Grad-CAM
layer_grad_cam = LayerGradCam(model, model.layer4)
attributions = layer_grad_cam.attribute(input_tensor, target=pred_label_idx)

# Visualize the heatmap
heatmap = attributions.squeeze().cpu().detach().numpy()
heatmap = np.maximum(heatmap, 0)  # Remove negative values
heatmap /= np.max(heatmap)  # Normalize to the range [0, 1]

# Display the original image and the heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Original image
original_image = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0))
ax1.imshow(original_image)
ax1.set_title("Original Image")
ax1.axis("off")

# Heatmap
heatmap_image = np.uint8(plt.cm.jet(heatmap)[:, :, :3] * 255)
im = ax2.imshow(heatmap_image)
ax2.set_title("Grad-CAM Heatmap")
ax2.axis("off")

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Importance', rotation=270, labelpad=20)

plt.tight_layout()
plt.show()