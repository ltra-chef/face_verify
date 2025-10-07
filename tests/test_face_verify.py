import subprocess
import sys
import os
import pytest

# Paths to test images
BASE_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, "tests/database")

PERSON1A = os.path.join(IMG_DIR, "barack_obama/barack1.jpg")
PERSON1B = os.path.join(IMG_DIR, "barack_obama/barack2.jpg")
PERSON2A = os.path.join(IMG_DIR, "joe_biden/biden1.jpg")

SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "face_verify.py"))


def run_cli(*args):
    """Run the face_verify.py CLI and return output as string"""
    cmd = [sys.executable, SCRIPT] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def test_same_person():
    """Verify that two images of the same person match"""
    output = run_cli("verify", PERSON1A, PERSON1B)
    assert "Verified: True" in output, f"Expected True, got output:\n{output}"


def test_different_person():
    """Verify that two different people do not match"""
    output = run_cli("verify", PERSON1A, PERSON2A)
    assert "Verified: False" in output, f"Expected False, got output:\n{output}"


@pytest.mark.parametrize("model", ["VGG-Face", "ArcFace"])
def test_with_different_models(model):
    """Test script with multiple models"""
    output = run_cli("verify", PERSON1A, PERSON1B, "--model", model)
    assert "Verified:" in output
    assert f"Model: {model}" in output
