import torch
import torchvision.models as models

# Load the pretrained ResNet model
model = models.resnet18(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Create an example input tensor
input_data = torch.randn(1, 3, 224, 224)

# Forward pass to get the intermediate outputs
outputs = []
def hook(module, input, output):
    outputs.append(output)

# Register the hook for each layer
for name, module in model.named_modules():
    module.register_forward_hook(hook)

# Forward pass
output = model(input_data)

# Print the intermediate outputs
for i, output in enumerate(outputs):
    print(f"Layer {i+1} output:")
    print(output)