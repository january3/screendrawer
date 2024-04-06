import subprocess
import pytest
from pathlib import Path


# Mapping of file extensions to expected magic numbers or starting strings
MAGIC_NUMBERS = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "svg": "<?xml",  # SVG is text, so this is a simplification
}

@pytest.mark.parametrize("fmt", ["png", "svg"])
def test_sd_conversion_binary(tmp_path, fmt):
    # Path to the test binary file
    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"
    
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)

    # Run the script with subprocess.run
    result = subprocess.run(['python3', 'sd_devel.py', str(file_in), str(file_out)], 
                            capture_output=True, text=True)
    
    # Check the script executed successfully
    assert result.returncode == 0, "Script failed to run successfully"

    assert file_out.exists(), "Output file was not created"

    # (i) Test file is not empty
    assert file_out.stat().st_size > 0, "File is empty"

    # (ii) Test file is a PNG file
    # Checking the magic number
    if fmt in ["png", "jpg"]:
        with open(file_out, 'rb') as f:
            file_start = f.read(8)  # Read more bytes to ensure covering the longer magic numbers
            expected_magic_number = MAGIC_NUMBERS[fmt]
            assert file_start.startswith(expected_magic_number), f"File format check failed for {fmt}"
    elif fmt == "svg":
        with open(file_out, 'r') as f:
            file_start = f.read(50)  # Read enough characters to reliably find the XML declaration or svg tag
            assert MAGIC_NUMBERS["svg"] in file_start, "File format check failed for SVG"

@pytest.mark.parametrize("fmt", ["png", "svg"])
def test_sd_conversion_binary2(tmp_path, fmt):
    # Path to the test binary file
    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"
    
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)

    # Run the script with subprocess.run
    result = subprocess.run(['python3', 'sd_devel.py', "-b 10", str(file_in), str(file_out)], 
                            capture_output=True, text=True)
    
    # Check the script executed successfully
    assert result.returncode == 0, "Script failed to run successfully"

    assert file_out.exists(), "Output file was not created"

    # (i) Test file is not empty
    assert file_out.stat().st_size > 0, "File is empty"

    if fmt in ["png", "jpg"]:
        with open(file_out, 'rb') as f:
            file_start = f.read(8)  # Read more bytes to ensure covering the longer magic numbers
            expected_magic_number = MAGIC_NUMBERS[fmt]
            assert file_start.startswith(expected_magic_number), f"File format check failed for {fmt}"
    elif fmt == "svg":
        with open(file_out, 'r') as f:
            file_start = f.read(50)  # Read enough characters to reliably find the XML declaration or svg tag
            assert MAGIC_NUMBERS["svg"] in file_start, "File format check failed for SVG"
