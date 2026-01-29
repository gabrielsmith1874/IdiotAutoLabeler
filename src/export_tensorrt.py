"""
Export EfficientNet-UNet to TensorRT Engine
============================================
Converts the PyTorch model to an optimized TensorRT engine for faster inference.

Usage:
    python src/export_tensorrt.py
    python src/export_tensorrt.py --input models/enemy_segmentation.pth --size 256
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

# Import model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_segmentation import EfficientNetUNet, STDC1Seg, CONFIG


def export_to_onnx(checkpoint_path: Path, onnx_path: Path, input_size: int = 256):
    """Export PyTorch model to ONNX format."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get encoder name from checkpoint
    saved_config = checkpoint.get("config", {})
    encoder_name = saved_config.get("encoder_name", CONFIG["encoder_name"])
    print(f"Encoder: {encoder_name}")

    # Load model based on encoder type
    if encoder_name == "stdc1":
        model = STDC1Seg(pretrained=False)
    else:
        model = EfficientNetUNet(encoder_name=encoder_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Exporting to ONNX: {onnx_path}")

    # Use dynamo=False for compatibility with TensorRT
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Fixed size for TensorRT optimization
        dynamo=False,  # Use legacy export for better TensorRT compatibility
    )

    # Remove any external data file that may have been created
    external_data_path = Path(str(onnx_path) + ".data")
    if external_data_path.exists():
        external_data_path.unlink()
        # Re-export with embedded weights
        import onnx
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save(onnx_model, str(onnx_path), save_as_external_data=False)

    print(f"ONNX export complete: {onnx_path}")
    return onnx_path


def export_to_tensorrt(onnx_path: Path, engine_path: Path, fp16: bool = True):
    """Convert ONNX model to TensorRT engine."""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"Building TensorRT engine from {onnx_path}...")
    print(f"FP16 mode: {fp16}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    print("Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved: {engine_path}")
    print(f"Engine size: {engine_path.stat().st_size / 1024 / 1024:.1f} MB")

    return engine_path


def verify_engine(engine_path: Path, input_size: int = 256):
    """Verify the TensorRT engine works correctly."""
    import tensorrt as trt
    import numpy as np

    print(f"\nVerifying TensorRT engine: {engine_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Get input/output info
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    print(f"Input: {input_name} {list(input_shape)}")
    print(f"Output: {output_name} {list(output_shape)}")

    # Allocate buffers
    import torch
    device = torch.device("cuda")

    input_tensor = torch.randn(1, 3, input_size, input_size, device=device, dtype=torch.float32)
    output_tensor = torch.zeros(1, 1, input_size, input_size, device=device, dtype=torch.float32)

    # Set tensor addresses
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    # Run inference
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    print(f"Output range: [{output_tensor.min().item():.3f}, {output_tensor.max().item():.3f}]")

    # Benchmark
    import time

    # Warmup
    for _ in range(10):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Benchmark
    n_runs = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / n_runs * 1000
    fps = 1000 / avg_time

    print(f"\nBenchmark ({n_runs} runs):")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Estimated FPS: {fps:.1f}")

    return fps


def main():
    parser = argparse.ArgumentParser(description="Export model to TensorRT")
    parser.add_argument("--input", "-i", type=str, default="models/enemy_segmentation.pth",
                       help="Input PyTorch checkpoint")
    parser.add_argument("--size", "-s", type=int, default=256,
                       help="Input size for the model")
    parser.add_argument("--fp32", action="store_true",
                       help="Use FP32 instead of FP16")
    args = parser.parse_args()

    checkpoint_path = Path(args.input)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Output paths
    base_name = checkpoint_path.stem
    onnx_path = checkpoint_path.parent / f"{base_name}_{args.size}.onnx"
    engine_path = checkpoint_path.parent / f"{base_name}_{args.size}.engine"

    print("=" * 60)
    print("TensorRT Export")
    print("=" * 60)
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Input size: {args.size}x{args.size}")
    print(f"Precision: {'FP32' if args.fp32 else 'FP16'}")
    print()

    # Step 1: Export to ONNX
    export_to_onnx(checkpoint_path, onnx_path, args.size)

    # Step 2: Convert to TensorRT
    export_to_tensorrt(onnx_path, engine_path, fp16=not args.fp32)

    # Step 3: Verify
    fps = verify_engine(engine_path, args.size)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print(f"TensorRT engine: {engine_path}")
    print(f"Expected FPS: {fps:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
