import torch
import onnx
import onnx_tensorrt.backend as backend
import tensorrt as trt
import time
import numpy as np

# Define a simple PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 512)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Export the PyTorch model to ONNX
model = MyModel()
input_shape = (1, 3, 32, 32)
input_names = ['input']
output_names = ['output']
dummy_input = torch.randn(input_shape)
torch.onnx.export(model, dummy_input, 'my_model.onnx', verbose=False, input_names=input_names, output_names=output_names)

# Load the ONNX model and create a TensorRT engine from it
model_onnx = onnx.load('my_model.onnx')
engine = backend.prepare(model_onnx, device='CUDA:0')

# Create a context for executing inference on the engine
context = engine.create_execution_context()

# Allocate device memory for input and output buffers
input_name = 'input'
output_name = 'output'
input_shape = (1, 3, 32, 32)
output_shape = (1, 10)
input_buf = trt.cuda.alloc_cuda_pinned_memory(trt.volume(input_shape) * trt.float32.itemsize)
output_buf = trt.cuda.alloc_cuda_pinned_memory(trt.volume(output_shape) * trt.float32.itemsize)

# Load the PyTorch model into memory and measure inference speed
model.load_state_dict(torch.load('my_model.pth'))
model.eval()
num_iterations = 1000
total_time = 0.0
with torch.no_grad():
    for i in range(num_iterations):
        start_time = time.time()
        input_data = torch.randn(input_shape)
        output_data = model(input_data)
        end_time = time.time()
        total_time += end_time - start_time
pytorch_fps = num_iterations / total_time
print(f"PyTorch FPS: {pytorch_fps:.2f}")

# Create a TensorRT engine from the ONNX model and measure inference speed
trt_engine = backend.prepare(model_onnx, device='CUDA:0')
num_iterations = 1000
total_time = 0.0
with torch.no_grad():
    for i in range(num_iterations):
        input_data = torch.randn(input_shape).cuda()
        start_time = time.time()
        output_data = trt_engine.run(input_data.cpu().numpy())[0]
        end_time = time.time()
        total_time += end_time - start_time
tensorrt_fps = num_iterations /total_time
tensorrt_fps = num_iterations / total_time
print(f"TensorRT FPS: {tensorrt_fps:.2f}")
print(f"Speedup: {tensorrt_fps/pytorch_fps:.2f}x")