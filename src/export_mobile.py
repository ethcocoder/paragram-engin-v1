import torch
import os
import argparse
from model import GenesisDecoder

def export_decoder_to_onnx(checkpoint_path="checkpoints/universal_genesis_core.pth", output_path="mobile_decoder.onnx", latent_channels=16):
    print(f"\n{'='*60}\n[*] PARADOX MOBILE EXPORT PIPELINE\n{'='*60}")
    
    # 1. Initialize Decoder
    decoder = GenesisDecoder(latent_channels=latent_channels)
    
    # 2. Load Weights if available
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading trained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # We only want the decoder weights
        decoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('decoder.'):
                decoder_state_dict[key.replace('decoder.', '')] = value
                
        decoder.load_state_dict(decoder_state_dict, strict=False)
    else:
        print("[!] Warning: Checkpoint not found. Exporting untrained model structure.")
        
    decoder.eval()
    
    # 3. Create Dummy Input 
    # (Latent space shape: B=1, C=latent_channels, H=16, W=16 for 256px image)
    # 256px / 16 (downsample factor) = 16
    dummy_latent = torch.randn(1, latent_channels, 16, 16)
    
    # 4. Export to ONNX
    print(f"[*] Exporting to {output_path}...")
    torch.onnx.export(
        decoder,                     
        dummy_latent,                
        output_path,                 
        export_params=True,          
        opset_version=12,            
        do_constant_folding=True,    
        input_names = ['latent_input'],   
        output_names = ['image_output'], 
        dynamic_axes={
            'latent_input' : {0 : 'batch_size', 2: 'height', 3: 'width'},
            'image_output' : {0 : 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    # Calculate File Size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"[+] SUCCESS: Model exported to {output_path}")
    print(f"[+] Mobile Decoder Size: {size_mb:.2f} MB")
    print("[+] Ready for iOS (CoreML) and Android (NNAPI) deployment.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox ONNX Exporter")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/universal_genesis_core.pth")
    parser.add_argument("--output", type=str, default="mobile_decoder.onnx")
    parser.add_argument("--channels", type=int, default=16)
    args = parser.parse_args()
    
    export_decoder_to_onnx(args.checkpoint, args.output, args.channels)
