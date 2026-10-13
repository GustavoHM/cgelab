import os
import pathlib
import sys

# The suite tests the source tree by default. CGELAB_TEST_INSTALLED=1 tests the
# installed package instead, which is how CI checks a wheel: an editable install
# can break silently (Python 3.14 on macOS skips some .pth files), and a test run
# against src/ would not notice.
if os.environ.get("CGELAB_TEST_INSTALLED") != "1":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
