import torch
import torchvision.models as models
import torch.quantization as quantization
import torch.onnx as onnx

# Load the pretrained ResNet model
model = models.resnet50(pretrained=True)

# Create an input tensor as an example
input_data = torch.randn(1, 3, 224, 224)

# Set the model to evaluation mode
model.eval()

# Define the quantization configuration
quant_config = quantization.default_dynamic_qconfig

# Prepare the model for dynamic quantization
quantized_model = quantization.quantize_dynamic(model, qconfig_spec=quant_config)

# Run the quantized model
output = quantized_model(input_data)

onnx.export(model, input_data, "resnet18_set11_quant.onnx", opset_version=11)

print("Quantized model output:", output)