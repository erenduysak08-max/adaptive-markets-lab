from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_offline_dashboard_renders_without_errors() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not list(app.exception)
    assert app.title[0].value == "Adaptive Markets Lab"
    assert len(app.dataframe) >= 3
    assert {"Exposure & trades", "Parameter heatmap", "Strategy code"}.issubset(
        {tab.label for tab in app.tabs}
    )


def test_pairs_dashboard_and_colour_table_render() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    strategy = next(item for item in app.selectbox if item.label == "Strategy")
    strategy.set_value("Pairs trading")
    app.run()

    assert not list(app.exception)
    assert app.metric[0].label == "Total return"
    assert any(tab.label == "Rolling regression" for tab in app.tabs)

    calculate = next(
        item for item in app.button if item.label == "Calculate colour-scale table"
    )
    calculate.click()
    app.run()

    assert not list(app.exception)
