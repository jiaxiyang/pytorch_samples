import torch
import torchvision.models as models
import numpy as np

# Load the pretrained ResNet model
model = models.resnet18(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Create an example input tensor
input_data = torch.randn(1, 3, 224, 224)

# Forward pass to get the intermediate outputs
outputs = {}
def hook(name):
    def fn(module, input, output):
        outputs[name] = output.cpu().detach().numpy()
    return fn

# Register the hook for each layer
for name, module in model.named_modules():
    module.register_forward_hook(hook(name))

# Forward pass
output = model(input_data)

# Save the intermediate outputs as binary files
for name, output in outputs.items():
    filename = f"{name}.bin"
    output.astype(np.float32).tofile(filename)
    print(f"Tensor {name} output saved to {filename}")