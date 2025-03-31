import torch

from inference_model import Net

def main():
    # Load the trained model's state dictionary
    model = Net()
    model.load_state_dict(torch.load("pytorch_model.pt"))
    
    # Set the model to evaluation mode
    model.eval()
    
    # Export the model to ONNX format
    dummy_input = torch.randn(1, 1, 28, 28)  # Example input tensor
    torch.onnx.export(model, dummy_input, "onnx_model_sm.onnx", verbose=True, opset_version=9)


if __name__ == '__main__':
    main()  # Run the main function
