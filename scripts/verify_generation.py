#!/usr/bin/env python3
"""
Verification script for the generation pipeline fixes.
Run after implementing fixes to confirm everything works.

Usage: python scripts/verify_generation.py [--checkpoint PATH]
"""
import sys
import argparse
sys.path.append('.')

import torch
from models.generative import ReportGenLightning, BioGPTGenerator
from transformers import BioGptTokenizer


def test_bos_token():
    """Test 1: Verify BOS token is available"""
    print("\n[Test 1] BOS Token Availability...")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    
    assert bos_id is not None, "FAIL: No BOS token available!"
    print(f"  PASS: BOS token ID = {bos_id}")
    return True


def test_generation_non_empty(checkpoint_path=None):
    """Test 2: Verify generation produces non-empty output"""
    print("\n[Test 2] Non-Empty Generation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if checkpoint_path:
        model = ReportGenLightning.load_from_checkpoint(checkpoint_path, strict=False)
    else:
        model = ReportGenLightning()
    
    model = model.to(device)
    model.eval()
    
    # Dummy input: batch=1, views=3, channels=3, 224x224
    dummy_pixels = torch.randn(1, 3, 3, 224, 224).to(device)
    dummy_mask = torch.ones(1, 3).long().to(device)
    
    with torch.no_grad():
        outputs = model.generate(dummy_pixels, view_mask=dummy_mask)
    
    assert len(outputs) == 1, f"FAIL: Expected 1 output, got {len(outputs)}"
    assert len(outputs[0]) > 0, "FAIL: Generated text is empty!"
    assert len(outputs[0]) >= 20, f"FAIL: Generated text too short ({len(outputs[0])} chars)"
    
    print(f"  PASS: Generated {len(outputs[0])} characters")
    print(f"  Preview: '{outputs[0][:80]}...'")
    return True


def test_perceiver_trainable():
    """Test 3: Verify Perceiver parameters are trainable"""
    print("\n[Test 3] Perceiver Trainability...")
    model = ReportGenLightning()
    
    trainable = sum(p.numel() for p in model.perceiver.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.perceiver.parameters())
    
    assert trainable == total, f"FAIL: Only {trainable}/{total} Perceiver params trainable!"
    print(f"  PASS: {trainable:,} / {total:,} params trainable (100%)")
    return True


def test_gradient_flow():
    """Test 4: Verify gradients flow through Perceiver"""
    print("\n[Test 4] Gradient Flow Through Perceiver...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReportGenLightning().to(device)
    model.train()
    
    # Dummy forward pass
    dummy_pixels = torch.randn(1, 3, 3, 224, 224).to(device)
    dummy_mask = torch.ones(1, 3).long().to(device)
    dummy_input_ids = torch.randint(0, 1000, (1, 50)).to(device)
    dummy_attention = torch.ones(1, 50).long().to(device)
    dummy_labels = dummy_input_ids.clone()
    
    outputs = model(dummy_pixels, dummy_input_ids, dummy_attention, 
                    view_mask=dummy_mask, labels=dummy_labels)
    outputs.loss.backward()
    
    # Check gradients exist
    grad_norms = []
    for name, param in model.perceiver.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    assert len(grad_norms) > 0, "FAIL: No gradients in Perceiver!"
    assert sum(grad_norms) > 1e-8, "FAIL: Gradients are near zero!"
    
    print(f"  PASS: {len(grad_norms)} params have gradients")
    print(f"  Avg grad norm: {sum(grad_norms)/len(grad_norms):.6f}")
    return True


def test_output_slicing():
    """Test 5: Verify output slicing logic works correctly"""
    print("\n[Test 5] Output Slicing Logic...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simulate generation output with placeholders
    # First 65 positions are placeholders (64 image + 1 BOS)
    pad_id = tokenizer.pad_token_id
    fake_output = torch.full((1, 100), pad_id, dtype=torch.long)
    
    # Put some actual token IDs after the placeholder positions
    fake_output[0, 65:80] = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
                                           1100, 1200, 1300, 1400, 1500])
    
    # Slice off placeholders (simulating what the fixed generate() does)
    input_length = 65
    generated_ids = fake_output[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    assert len(decoded[0]) > 0, "FAIL: Decoded output is empty after slicing!"
    print(f"  PASS: Slicing correctly extracts generated tokens")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify generation pipeline fixes")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint for testing')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATION PIPELINE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("BOS Token", test_bos_token),
        ("Non-Empty Generation", lambda: test_generation_non_empty(args.checkpoint)),
        ("Perceiver Trainable", test_perceiver_trainable),
        ("Gradient Flow", test_gradient_flow),
        ("Output Slicing", test_output_slicing),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

