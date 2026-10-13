"""Regression tests for blocking, reporting, audit signatures, and CLI imports."""
import numpy as np
import pandas as pd
import pytest

from agents.base import AuditTrail
from cli import _parse_boolean
from record_linkage.linker import run_linkage
from record_linkage.model import FellegiSunterModel


def test_audit_signature_validation_detects_all_mutations():
    chain = AuditTrail(secret_key="local-test-key")
    chain.log("test", "operator", "created", {"count": 1})
    chain.log("test", "operator", "updated", {"count": 2})
    assert chain.verify_integrity()
    chain.get_trail()[0]["event_type"] = "modified"
    assert chain.verify_integrity()  # returned copy cannot mutate the ledger
    chain.logs[1]["event_type"] = "modified"
    assert not chain.verify_integrity()


def test_csv_booleans_are_parsed_explicitly():
    assert not _parse_boolean("False")
    assert not _parse_boolean("0")
    assert _parse_boolean("YES")
    with pytest.raises(ValueError):
        _parse_boolean("perhaps")


def test_empty_candidate_report_retains_export_columns(tmp_path):
    file = tmp_path / "unrelated.csv"
    pd.DataFrame([
        {"first_name": "Alice", "last_name": "Jones", "dob": "1990-03-10"},
        {"first_name": "Bob", "last_name": "Smith", "dob": "1999-05-11"},
    ]).to_csv(file, index=False)
    result = run_linkage([str(file)])
    assert result["n_candidate_pairs"] == 0
    assert "classification" in result["report"].columns
    assert "first_name_similarity" in result["report"].columns
    assert "classification" in result["report"].to_csv(index=False)


def test_extra_blocking_columns_are_normalized(tmp_path):
    file = tmp_path / "input.csv"
    pd.DataFrame([
        {"first_name": "Ann", "last_name": "Lee", "dob": "2000-05-01", "site": " North "},
        {"first_name": "Ana", "last_name": "Lee", "dob": "2000-05-01", "site": "north"},
    ]).to_csv(file, index=False)
    result = run_linkage([str(file)], field_specs_raw="first_name:jw,last_name:jw", blocking_spec_raw="site:exact")
    assert result["n_candidate_pairs"] == 1


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        FellegiSunterModel.with_default_probabilities(["name"], m_default=np.nan)
    with pytest.raises(ValueError):
        FellegiSunterModel.with_default_probabilities(["name"], m_default=1.1)
    model = FellegiSunterModel.with_default_probabilities(["name"])
    with pytest.raises(ValueError):
        model.fit_em(np.array([[1.]]), max_iter=0)
