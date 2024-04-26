import subprocess
import pytest
from pathlib import Path

EXECUTABLE = "src/sd"

# Mapping of file extensions to expected magic numbers or starting strings
MAGIC_NUMBERS = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "pdf": b"%PDF",
    "svg": "<?xml",  # SVG is text, so this is a simplification
}

# Path to the test binary file
TEST_DATA_DIR = Path(__file__).parent / "data"
FILE_IN = TEST_DATA_DIR / "foobar.sdrw"
 
def _check_file_format(cmdline, file_out, fmt):
    """
    Check that the file at file_path is of the expected format

    Arguments:
    cmdline -- the command line to run the script
    file_out -- the path to the output file
    fmt -- the expected format of the file
    """

    result = subprocess.run(cmdline,
                            capture_output=True, text=True)

    # Check the script executed successfully
    if result.returncode != 0:
        print("Standard Error Output:", result.stderr)
        print("Standard Output:", result.stdout)
        assert False, f"Script failed to run successfully. Return code: {result.returncode}"


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
    """Test the script with a valid input file and output format"""
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)
    cmdline = ['python3', EXECUTABLE, str(FILE_IN), str(file_out)] 
    _check_file_format(cmdline, file_out, fmt)
    
@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_sd_conversion_with_boundary(tmp_path, fmt):
    """Test -b option"""
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)
    cmdline = ['python3', EXECUTABLE, "-b 10", str(FILE_IN), str(file_out)]
    _check_file_format(cmdline, file_out, fmt)

@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_sd_conversion_explicit_format(tmp_path, fmt):
    """Test -c option"""
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)
    cmdline = ["python3", EXECUTABLE, "-c", fmt, "-o", str(file_out), str(FILE_IN)]
    _check_file_format(cmdline, file_out, fmt)


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_sd_conversion_with_page_no(tmp_path, fmt):
    """Test -p option"""
    # Define the output file path
    file_out = tmp_path / ("output." + fmt)
    cmdline = ["python3", EXECUTABLE, "-p", "2", "-c", fmt, "-o", str(file_out), str(FILE_IN)]
    _check_file_format(cmdline, file_out, fmt)

def test_sd_conversion_to_multipage_pdf(tmp_path):
    """Test multipage pdf conversion"""
    # Define the output file path
    file_out = tmp_path / ("output.pdf")
    cmdline = ["python3", EXECUTABLE, str(FILE_IN), str(file_out)]
    _check_file_format(cmdline, file_out, "pdf")

