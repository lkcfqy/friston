import torch
import torch.nn.functional as F

def test_mamba_import():
    print("Testing Mamba Import...")
    try:
        from mamba_ssm import Mamba
        print("✅ Mamba import successful")
    except ImportError as e:
        print(f"❌ Mamba import suspected failure: {e}")
        return

    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Mamba typically requires CUDA.")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")

    try:
        # Define a simple Mamba layer
        batch, length, dim = 2, 64, 16
        model = Mamba(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).cuda()
        
        x = torch.randn(batch, length, dim).cuda()
        y = model(x)
        
        print(f"✅ Forward pass successful. Output shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ Mamba runtime error: {e}")

if __name__ == "__main__":
    test_mamba_import()
