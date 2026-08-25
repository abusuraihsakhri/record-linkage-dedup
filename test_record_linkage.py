"""Tests for the Fellegi-Sunter record linkage pipeline.

Plain assert-based tests, run with: python test_record_linkage.py
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd

from record_linkage.similarity import jaro, jaro_winkler
from record_linkage.model import FellegiSunterModel
from record_linkage.linker import run_linkage


def test_jaro_identical_strings_score_one():
    assert jaro("smith", "smith") == 1.0
    assert jaro_winkler("smith", "smith") == 1.0


def test_jaro_winkler_close_typo_scores_high_and_boosts_shared_prefix():
    # classic Winkler example: MARTHA vs MARHTA
    score = jaro_winkler("martha", "marhta")
    assert 0.96 < score <= 1.0

    # shared prefix should push the winkler score above plain jaro
    plain = jaro("dwayne", "duane")
    winkler = jaro_winkler("dwayne", "duane")
    assert winkler >= plain

    # completely different strings score low
    assert jaro_winkler("abcdef", "zzzzzz") < 0.2


def test_fellegi_sunter_weights_reward_agreement_on_discriminating_fields():
    # A field that rarely agrees by chance (low u) but agrees often for
    # true matches (high m) should get a large positive agreement weight,
    # and a correspondingly large negative disagreement weight.
    model = FellegiSunterModel(
        field_names=["first_name", "last_name", "dob"],
        m_prob=np.array([0.9, 0.9, 0.95]),
        u_prob=np.array([0.1, 0.05, 0.01]),
        prior=0.05,
    )
    agree_w = model.agreement_weights()
    disagree_w = model.disagreement_weights()

    assert np.all(agree_w > 0)
    assert np.all(disagree_w < 0)
    # dob has the lowest u (rarest chance agreement) so it should carry the
    # single largest agreement weight of the three fields.
    assert agree_w[2] == max(agree_w)

    all_agree = model.score_pairs(np.array([[1, 1, 1]]))[0]
    all_disagree = model.score_pairs(np.array([[0, 0, 0]]))[0]
    assert all_agree > all_disagree
    assert math.isclose(all_agree, agree_w.sum())
    assert math.isclose(all_disagree, disagree_w.sum())


def test_em_recovers_separated_match_and_nonmatch_clusters():
    # Build a synthetic agreement matrix with an obvious mixture: a small
    # cluster of "true matches" that mostly agree on 3 fields, and a large
    # cluster of "true non-matches" that mostly disagree. EM should recover
    # m-probabilities well above u-probabilities after fitting.
    rng = np.random.default_rng(0)
    n_matches, n_nonmatches = 40, 400
    true_m = np.array([0.95, 0.9, 0.97])
    true_u = np.array([0.15, 0.1, 0.02])

    match_rows = (rng.random((n_matches, 3)) < true_m).astype(float)
    nonmatch_rows = (rng.random((n_nonmatches, 3)) < true_u).astype(float)
    agreement = np.vstack([match_rows, nonmatch_rows])
    rng.shuffle(agreement)

    model = FellegiSunterModel.with_default_probabilities(
        ["first_name", "last_name", "dob"], m_default=0.9, u_default=0.1, prior=0.1
    )
    info = model.fit_em(agreement, max_iter=50)

    assert info["iterations"] >= 1
    # log-likelihood should be non-decreasing (monotone EM) after the first step
    ll = info["log_likelihood"]
    assert all(b >= a - 1e-6 for a, b in zip(ll, ll[1:]))
    # fitted m should exceed fitted u on every field, in roughly the right ballpark
    assert np.all(model.m_prob > model.u_prob)
    for est, truth in zip(model.m_prob, true_m):
        assert abs(est - truth) < 0.15


def test_classify_buckets_scores_correctly():
    scores = np.array([10.0, 5.0, -3.0])
    labels = FellegiSunterModel.classify(scores, upper=8.0, lower=2.0)
    assert labels == ["match", "possible_match", "non_match"]


def _write_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_end_to_end_linkage_on_two_small_datasets():
    columns = ["patient_id", "first_name", "last_name", "dob"]
    dataset_a = [
        [1, "John", "Smith", "1980-05-14"],
        [2, "Linda", "Davis", "1965-07-30"],
    ]
    dataset_b = [
        [101, "Jon", "Smith", "1980-05-14"],   # near-duplicate of record 1
        [102, "Susan", "Miller", "1972-09-19"],  # unrelated
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "a.csv")
        path_b = os.path.join(tmp, "b.csv")
        _write_csv(path_a, dataset_a, columns)
        _write_csv(path_b, dataset_b, columns)

        result = run_linkage(
            paths=[path_a, path_b],
            field_specs_raw="first_name:jw,last_name:jw,dob:near",
            blocking_spec_raw="dob:year,first_name:initial",
        )

        report = result["report"]
        assert result["n_records"] == 4
        # John Smith / Jon Smith share a blocking key (1980, 'J'); the other
        # two records land in different blocks and are never compared.
        assert result["n_candidate_pairs"] == 1
        assert len(report) == 1

        row = report.iloc[0]
        assert row["first_name_a"] in ("john", "jon")
        assert row["last_name_agree"] == True  # noqa: E712 - exact string match
        assert row["dob_agree"] == True  # noqa: E712
        assert row["classification"] == "match"


def test_end_to_end_dedup_within_single_dataset():
    columns = ["patient_id", "first_name", "last_name", "dob"]
    rows = [
        [1, "Robert", "Brown", "1990-01-23"],
        [2, "Robert", "Braun", "1990-01-23"],  # likely duplicate, last-name typo
        [3, "Alice", "Nguyen", "1999-12-01"],  # unrelated, different block
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "records.csv")
        _write_csv(path, rows, columns)

        result = run_linkage(
            paths=[path],
            field_specs_raw="first_name:jw,last_name:jw,dob:near",
            blocking_spec_raw="dob:year,first_name:initial",
        )

        report = result["report"]
        assert result["n_candidate_pairs"] == 1
        assert report.iloc[0]["classification"] in ("match", "possible_match")
        assert report.iloc[0]["first_name_agree"] == True  # noqa: E712


if __name__ == "__main__":
    tests = [
        test_jaro_identical_strings_score_one,
        test_jaro_winkler_close_typo_scores_high_and_boosts_shared_prefix,
        test_fellegi_sunter_weights_reward_agreement_on_discriminating_fields,
        test_em_recovers_separated_match_and_nonmatch_clusters,
        test_classify_buckets_scores_correctly,
        test_end_to_end_linkage_on_two_small_datasets,
        test_end_to_end_dedup_within_single_dataset,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
