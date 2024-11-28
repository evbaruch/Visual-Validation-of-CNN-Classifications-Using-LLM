import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from captum.attr import GradientShap

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
image_path = r'c:\cats_and_dogs_filtered\train\cats\cat.101.jpg'
img = Image.open(image_path)
transformed_img = transform(img)
input_tensor = transform_normalize(transformed_img).unsqueeze(0)

# Get the model prediction for the original image
output = model(input_tensor)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = 'cat'  # Assign the class name based on your dataset
print('Predicted original:', predicted_label, '(', prediction_score.squeeze().item(), ')')

# Initialize GradientShap
gradient_shap = GradientShap(model)

# Compute attributions with GradientShap
baselines = torch.zeros_like(input_tensor)
attributions_gs = gradient_shap.attribute(input_tensor, baselines=baselines, target=pred_label_idx)

# Normalize attributions to [0, 1]
attributions_gs = attributions_gs.squeeze().cpu().detach().numpy()
attributions_gs = np.abs(attributions_gs)  # Take absolute values
attributions_gs = np.max(attributions_gs, axis=0)  # Take maximum along the channels
attributions_gs /= np.max(attributions_gs)  # Normalize to the range [0, 1]

# Convert the tensor image to numpy and transpose for display
img_np = transformed_img.permute(1, 2, 0).cpu().detach().numpy()

# Threshold for displaying pixels
threshold = 0.1

# Create mask for pixels below threshold
mask_below_threshold = img_np.copy()
mask_below_threshold[attributions_gs < threshold] = 0

# Convert masked image back to tensor for prediction
masked_tensor = torch.tensor(mask_below_threshold.transpose(2, 0, 1), dtype=torch.float32)
masked_tensor = transform_normalize(masked_tensor).unsqueeze(0)

# Get the model prediction for the masked image
masked_output = model(masked_tensor)
masked_output = F.softmax(masked_output, dim=1)
masked_prediction_score, masked_pred_label_idx = torch.topk(masked_output, 1)
masked_pred_label_idx.squeeze_()
masked_predicted_label = 'cat'  # Assign the class name based on your dataset
print('Predicted masked:', masked_predicted_label, '(', masked_prediction_score.squeeze().item(), ')')

# Calculate accuracy for original and masked images
original_accuracy = 1 if pred_label_idx.item() == pred_label_idx.item() else 0
masked_accuracy = 1 if pred_label_idx.item() == masked_pred_label_idx.item() else 0

# Visualize the GradientShap heatmap alongside the original image and masked image
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Display original image on the left
axs[0].imshow(img_np)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display heatmap in the middle
axs[1].imshow(attributions_gs, cmap='jet')
axs[1].set_title('Gradient Magnitudes Heatmap')
axs[1].axis('off')
fig.colorbar(axs[1].imshow(attributions_gs, cmap='jet'), ax=axs[1], orientation='vertical')

# Display masked image on the right
axs[2].imshow(mask_below_threshold)
axs[2].set_title(f'Pixels Below Threshold ({threshold})')
axs[2].axis('off')

plt.tight_layout()
plt.show()
