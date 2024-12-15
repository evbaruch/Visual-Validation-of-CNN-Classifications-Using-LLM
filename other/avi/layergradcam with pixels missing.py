import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
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
image_path = r'c:\cats_and_dogs_filtered\train\cats\cat.5.jpg'
img = Image.open(image_path)
transformed_img = transform(img)
input_tensor = transform_normalize(transformed_img).unsqueeze(0)

# Get the model prediction for the original image
output = model(input_tensor)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label_original = 'cat'
print('Predicted original:', predicted_label_original, '(', prediction_score.squeeze().item(), ')')

# Apply Grad-CAM
layer_grad_cam = LayerGradCam(model, model.layer4)
attributions = layer_grad_cam.attribute(input_tensor, target=pred_label_idx)

# Visualize the heatmap
heatmap = attributions.squeeze().cpu().detach().numpy()
heatmap = np.maximum(heatmap, 0)  # Remove negative values
heatmap /= np.max(heatmap)  # Normalize to the range [0, 1]

# Resize heatmap to match original image size
heatmap_resized = np.array(Image.fromarray(heatmap).resize((224, 224), Image.LANCZOS))

# Apply threshold to the resized heatmap
threshold = 0.17
thresholded_heatmap = np.where(heatmap_resized > threshold, heatmap_resized, 0)

# Create a mask of the thresholded heatmap
mask = thresholded_heatmap > 0

# Convert transformed image to numpy
original_image = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0))

# Apply the mask to the original image
masked_image = original_image.copy()
masked_image[~mask] = 0

# Convert masked image back to tensor and normalize
masked_image_tensor = torch.tensor(masked_image).permute(2, 0, 1).unsqueeze(0)
masked_image_tensor = transform_normalize(masked_image_tensor)

# Get prediction for masked image
output_masked = model(masked_image_tensor)
output_masked = F.softmax(output_masked, dim=1)
prediction_score_masked, pred_label_idx_masked = torch.topk(output_masked, 1)
pred_label_idx_masked.squeeze_()
predicted_label_masked = 'cat'
print('Predicted masked:', predicted_label_masked, '(', prediction_score_masked.squeeze().item(), ')')

# Display the original image, the heatmap, and the masked image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Original image
ax1.imshow(original_image)
ax1.set_title("Original Image")
ax1.axis("off")

# Heatmap
heatmap_image = np.uint8(plt.cm.jet(heatmap_resized)[:, :, :3] * 255)
ax2.imshow(original_image)
ax2.imshow(heatmap_image, alpha=0.6)
ax2.set_title("Grad-CAM Heatmap")
ax2.axis("off")

# Thresholded original image
ax3.imshow(masked_image)
ax3.set_title("Thresholded Original Image")
ax3.axis("off")

plt.tight_layout()
plt.show()
