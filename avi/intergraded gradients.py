import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from captum.attr import IntegratedGradients, visualization as viz

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

# Get the model prediction
output = model(input_tensor)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = 'cat'
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

# Initialize Integrated Gradients
integrated_gradients = IntegratedGradients(model)

# Compute attributions with Integrated Gradients
attributions_ig = integrated_gradients.attribute(input_tensor, target=pred_label_idx)

# Normalize attributions to [0, 1]
attributions_ig = attributions_ig.squeeze().cpu().detach().numpy()
attributions_ig = np.abs(attributions_ig)  # Take absolute values
attributions_ig = np.amax(attributions_ig, axis=0)  # Take maximum along the channels

# Visualize the Integrated Gradients heatmap alongside the original image
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display original image on the left
axs[0].imshow(transformed_img.permute(1, 2, 0))  # permute to convert tensor to (H, W, C) for display
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display heatmap on the right
heatmap = axs[1].imshow(attributions_ig, cmap='jet')
axs[1].set_title('Integrated Gradients Heatmap')
axs[1].axis('off')
fig.colorbar(heatmap, ax=axs[1], orientation='vertical')

plt.tight_layout()
plt.show()
