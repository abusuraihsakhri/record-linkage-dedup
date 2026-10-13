"""CLI spreadsheet-injection defense."""
from record_linkage.cli import _spreadsheet_safe

def test_export_neutralizes_untrusted_formulas():
    for payload in ["=SUM(1,2)", "+A1", "-CMD()", "@SUM(1)", "  =1+1"]:
        assert _spreadsheet_safe(payload).startswith("'")
    assert _spreadsheet_safe("smith") == "smith"
    assert _spreadsheet_safe(2.0) == 2.0
