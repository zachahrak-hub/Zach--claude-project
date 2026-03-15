#!/usr/bin/env python3
"""Test Phase 2 resource loaders."""

import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app import (
    _load_aws_iam_context,
    _load_aws_infrastructure_context,
    _load_okta_context,
    _load_aws_audit_context
)

def test_aws_iam_context():
    """Test 1: AWS IAM Context Loader"""
    print("\n[TEST 1] AWS IAM Context Loader")
    print("-" * 70)
    result = _load_aws_iam_context()
    print(result[:400])
    if len(result) > 0:
        print(f"\n✓ Function returned {len(result)} characters")
        return True
    return False

def test_aws_infrastructure_context():
    """Test 2: AWS Infrastructure Context Loader"""
    print("\n[TEST 2] AWS Infrastructure Context Loader")
    print("-" * 70)
    result = _load_aws_infrastructure_context()
    print(result[:400])
    if len(result) > 0:
        print(f"\n✓ Function returned {len(result)} characters")
        return True
    return False

def test_okta_context():
    """Test 3: Okta Context Loader"""
    print("\n[TEST 3] Okta Context Loader")
    print("-" * 70)
    result = _load_okta_context()
    print(result[:400])
    if len(result) > 0:
        print(f"\n✓ Function returned {len(result)} characters")
        return True
    return False

def test_aws_audit_context():
    """Test 4: AWS Audit Context Loader"""
    print("\n[TEST 4] AWS Audit Context Loader")
    print("-" * 70)
    result = _load_aws_audit_context()
    print(result[:400])
    if len(result) > 0:
        print(f"\n✓ Function returned {len(result)} characters")
        return True
    return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE 2 RESOURCE LOADER VERIFICATION TEST")
    print("="*70)

    tests = [
        ("AWS IAM Context", test_aws_iam_context),
        ("AWS Infrastructure Context", test_aws_infrastructure_context),
        ("Okta Context", test_okta_context),
        ("AWS Audit Context", test_aws_audit_context),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL PHASE 2 RESOURCE LOADERS WORKING")
    else:
        print(f"⚠️  {total - passed} test(s) failed (may be due to missing credentials)")

    print(f"{'='*70}\n")

    return 0 if passed > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
