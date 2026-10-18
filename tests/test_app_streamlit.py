"""
Headless test of the REAL dashboard with Streamlit's own test runner (streamlit.testing.v1.AppTest).
Run on your computer after training:   python tests/test_app_streamlit.py     (or: pytest tests/test_app_streamlit.py)

NOTE: this file could not be executed in the environment where the project was built (Streamlit was not installable there),
so it has not been run by the author. If it fails, the error text will point to the failing page.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest


def test_every_page_starts_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    pages = at.sidebar.radio[0].options
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"page '{page}' raised: {at.exception}"


if __name__ == "__main__":
    test_every_page_starts_without_exceptions()
    print("Dashboard test passed: all pages started without exceptions.")
