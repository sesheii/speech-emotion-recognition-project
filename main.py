import torch


def test_pytorch_cuda():
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")

    if cuda_available:
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

        try:
            x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")

            y = x * 2
            print(f"GPU Calculation Result: {y.cpu().numpy()}")
        except Exception as e:
            print(f"\nError during GPU operation: {e}")
    else:
        print("\nStatus: PyTorch can see your CPU, but cannot find your GPU.")
        print("Tip: Check if you installed the 'cuXXX' version of torch.")


def main():
    print("Hello from speech-emotion-recognition-project!")


if __name__ == "__main__":
    main()
    test_pytorch_cuda()
