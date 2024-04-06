import subprocess
import pytest
from pathlib import Path


# Mapping of file extensions to expected magic numbers or starting strings
MAGIC_NUMBERS = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "pdf": b"%PDF",
    "svg": "<?xml",  # SVG is text, so this is a simplification
}

def _check_file_format(file_out, fmt):
    """
    Check that the file at file_path is of the expected format

    Arguments:
    file_out: Path to the file to check
    fmt: Expected format of the file (one of "png", "jpg", "svg", "pdf")
    """
    # check that file exists
    assert file_out.exists(), "Output file was not created"

    # check that the file is not empty
    assert file_out.stat().st_size > 0, "File is empty"

    if fmt in ["png", "jpg", "pdf"]:
        with open(file_out, 'rb') as f:
            file_start = f.read(8)  # Read more bytes to ensure covering the longer magic numbers
            expected_magic_number = MAGIC_NUMBERS[fmt]
            assert file_start.startswith(expected_magic_number), f"File format check failed for {fmt}"
    elif fmt == "svg":
        with open(file_out, 'r') as f:
            file_start = f.read(50)  # Read enough characters to reliably find the XML declaration or svg tag
            assert MAGIC_NUMBERS["svg"] in file_start, "File format check failed for SVG"


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_sd_conversion(tmp_path, fmt):
    # Path to the test binary file
    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"
    
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)

    # Run the script with subprocess.run
    result = subprocess.run(['python3', 'sd_devel.py', str(file_in), str(file_out)], 
                            capture_output=True, text=True)
    _check_file_format(file_out, fmt)
    
@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_sd_conversion_with_boundary(tmp_path, fmt):
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
    _check_file_format(file_out, fmt)

@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_sd_conversion_explicit_format(tmp_path, fmt):
    # Path to the test binary file
    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"
    
    # Define the output file path
    file_out = tmp_path / "output"

    # Run the script with subprocess.run
    result = subprocess.run(["python3", "sd_devel.py", "-c", fmt, "-o", str(file_out), str(file_in)],
                            capture_output=True, text=True)
    
    # Check the script executed successfully
    if result.returncode != 0:
        print("Standard Error Output:", result.stderr)
        print("Standard Output:", result.stdout)
        assert False, f"Script failed to run successfully. Return code: {result.returncode}"

    _check_file_format(file_out, fmt)
