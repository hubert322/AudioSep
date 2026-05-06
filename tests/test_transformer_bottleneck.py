"""Local CPU-only tests for the TransformerBottleneck integration.

Run from the AudioSep project root:
    python -m tests.test_transformer_bottleneck

These tests validate correctness without GPU hardware.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# Fix for PyTorch 2.6+ weights_only=True default
# Instead of allowlisting individual types, we monkeypatch torch.load 
# to default to weights_only=False for these tests.
import torch.serialization
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load



def test_bottleneck_shape():
    """Test 1: TransformerBottleneck produces correct output shape."""
    from models.transformer_bottleneck import TransformerBottleneck

    bottleneck = TransformerBottleneck(
        audio_channels=384,
        text_embed_dim=512,
        d_model=384,
        nhead=8,
        num_layers=2,  # fewer layers for fast CPU test
        dim_feedforward=1536,
        dropout=0.0,
    )
    bottleneck.eval()

    x = torch.randn(2, 384, 16, 16)
    condition = torch.randn(2, 512)

    with torch.no_grad():
        out = bottleneck(x, condition)

    assert out.shape == (2, 384, 16, 16), f"Expected (2, 384, 16, 16), got {out.shape}"
    print("✓ test_bottleneck_shape passed")


def test_bottleneck_different_spatial():
    """Test 1b: TransformerBottleneck handles non-square spatial dims."""
    from models.transformer_bottleneck import TransformerBottleneck

    bottleneck = TransformerBottleneck(
        audio_channels=384,
        text_embed_dim=512,
        d_model=384,
        nhead=8,
        num_layers=2,
        dim_feedforward=1536,
    )
    bottleneck.eval()

    # Non-square spatial dims (T/32=16, F/64=8)
    x = torch.randn(2, 384, 16, 8)
    condition = torch.randn(2, 512)

    with torch.no_grad():
        out = bottleneck(x, condition)

    assert out.shape == (2, 384, 16, 8), f"Expected (2, 384, 16, 8), got {out.shape}"
    print("✓ test_bottleneck_different_spatial passed")


def test_full_forward_pass():
    """Test 2: Full ResUNet30 forward pass with transformer bottleneck."""
    from models.resunet import ResUNet30

    model = ResUNet30(
        input_channels=1,
        output_channels=1,
        condition_size=512,
        use_transformer_bottleneck=True,
        transformer_config={
            'text_embed_dim': 512,
            'd_model': 384,
            'nhead': 8,
            'num_layers': 2,
            'dim_feedforward': 1536,
            'dropout': 0.0,
        },
    )
    model.eval()

    # 5 seconds at 32kHz = 160000 samples
    segment_samples = 160000
    batch_size = 2

    input_dict = {
        'mixture': torch.randn(batch_size, 1, segment_samples),
        'condition': torch.randn(batch_size, 512),
    }

    with torch.no_grad():
        output_dict = model(input_dict)

    waveform = output_dict['waveform']
    assert waveform.shape == (batch_size, 1, segment_samples), \
        f"Expected ({batch_size}, 1, {segment_samples}), got {waveform.shape}"
    print("✓ test_full_forward_pass passed")


def test_backward_pass():
    """Test 3: Gradients flow through the transformer bottleneck."""
    from models.resunet import ResUNet30

    model = ResUNet30(
        input_channels=1,
        output_channels=1,
        condition_size=512,
        use_transformer_bottleneck=True,
        transformer_config={
            'text_embed_dim': 512,
            'd_model': 384,
            'nhead': 8,
            'num_layers': 2,
            'dim_feedforward': 1536,
            'dropout': 0.0,
        },
    )
    model.train()

    segment_samples = 32000  # 1 second, smaller for faster backward
    batch_size = 2

    input_dict = {
        'mixture': torch.randn(batch_size, 1, segment_samples),
        'condition': torch.randn(batch_size, 512),
    }

    output_dict = model(input_dict)
    waveform = output_dict['waveform']

    # Simple L1 loss against zeros
    loss = torch.mean(torch.abs(waveform))
    loss.backward()

    # Check gradients exist on transformer bottleneck parameters
    has_grad = False
    for name, param in model.named_parameters():
        if 'transformer_bottleneck' in name and param.grad is not None:
            has_grad = True
            assert param.grad.abs().sum() > 0, \
                f"Parameter {name} has zero gradients"

    assert has_grad, "No gradients found on transformer_bottleneck parameters"
    print("✓ test_backward_pass passed")


def test_film_compatibility():
    """Test 4: get_film_meta excludes conv_block7a when transformer is active."""
    from models.resunet import ResUNet30, ResUNet30_Base, get_film_meta

    # With transformer bottleneck
    base_transformer = ResUNet30_Base(
        input_channels=1,
        output_channels=1,
        use_transformer_bottleneck=True,
        transformer_config={'num_layers': 2},
    )
    film_meta_transformer = get_film_meta(base_transformer)

    assert 'conv_block7a' not in film_meta_transformer, \
        "conv_block7a should not appear in film_meta when transformer bottleneck is active"

    # Without transformer bottleneck (baseline)
    base_original = ResUNet30_Base(
        input_channels=1,
        output_channels=1,
        use_transformer_bottleneck=False,
    )
    film_meta_original = get_film_meta(base_original)

    assert 'conv_block7a' in film_meta_original, \
        "conv_block7a should appear in film_meta when using original bottleneck"

    print("✓ test_film_compatibility passed")


def test_config_loading():
    """Test 5: audiosep_transformer.yaml loads and constructs model."""
    from utils import parse_yaml
    from models.audiosep import get_model_class

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'audiosep_transformer.yaml'
    )

    configs = parse_yaml(config_path)

    # Verify transformer config keys exist
    assert configs['model']['use_transformer_bottleneck'] == True, \
        "use_transformer_bottleneck should be True"
    assert 'transformer_config' in configs['model'], \
        "transformer_config should be present"

    tc = configs['model']['transformer_config']
    assert tc['d_model'] == 384
    assert tc['nhead'] == 8
    assert tc['num_layers'] == 4
    assert tc['text_embed_dim'] == 512

    # Construct model from config
    Model = get_model_class(configs['model']['model_type'])
    model = Model(
        input_channels=configs['model']['input_channels'],
        output_channels=configs['model']['output_channels'],
        condition_size=configs['model']['condition_size'],
        use_transformer_bottleneck=configs['model']['use_transformer_bottleneck'],
        transformer_config=configs['model']['transformer_config'],
    )

    assert hasattr(model.base, 'transformer_bottleneck'), \
        "Model should have transformer_bottleneck attribute"
    assert not hasattr(model.base, 'conv_block7a'), \
        "Model should not have conv_block7a when transformer is active"

    print("✓ test_config_loading passed")


def test_baseline_parity():
    """Test 6: use_transformer_bottleneck=False produces original model structure."""
    from models.resunet import ResUNet30

    model = ResUNet30(
        input_channels=1,
        output_channels=1,
        condition_size=512,
        use_transformer_bottleneck=False,
    )

    assert hasattr(model.base, 'conv_block7a'), \
        "Baseline model should have conv_block7a"
    assert not hasattr(model.base, 'transformer_bottleneck'), \
        "Baseline model should not have transformer_bottleneck"

    # Verify forward pass works
    model.eval()
    segment_samples = 32000
    input_dict = {
        'mixture': torch.randn(1, 1, segment_samples),
        'condition': torch.randn(1, 512),
    }

    with torch.no_grad():
        output_dict = model(input_dict)

    assert output_dict['waveform'].shape == (1, 1, segment_samples), \
        "Baseline forward pass should produce correct shape"

    print("✓ test_baseline_parity passed")


if __name__ == '__main__':
    print("=" * 60)
    print("TransformerBottleneck Local Tests (CPU)")
    print("=" * 60)
    print()

    tests = [
        test_bottleneck_shape,
        test_bottleneck_different_spatial,
        test_full_forward_pass,
        test_backward_pass,
        test_film_compatibility,
        test_config_loading,
        test_baseline_parity,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
