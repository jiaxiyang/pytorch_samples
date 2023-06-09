import torch
import torchvision.models as models
import torch.onnx as onnx

# 加载预训练模型
model = models.resnet18(pretrained=True)

# 创建一个输入张量作为示例
input_data = torch.randn(1, 3, 224, 224)

# 设置模型为推理模式
model.eval()

# 将模型和输入张量转换为ONNX格式
onnx_path = "model.onnx"
onnx.export(model, input_data, onnx_path)
onnx.export(model, input_data, "resnet18_set11.onnx", opset_version=11)

print("模型已成功转换为ONNX格式并保存在:", onnx_path)
