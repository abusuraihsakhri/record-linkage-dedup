"""Fellegi-Sunter probabilistic record linkage model.

Implements the classic Fellegi & Sunter (1969) framework:

* Each candidate record pair is reduced to a binary agreement vector,
  one entry per compared field.
* Each field has an m-probability (chance the field agrees given the
  pair is a *true match*) and a u-probability (chance the field agrees
  given the pair is a *true non-match*).
* The field's contribution to the match score is a log-likelihood-ratio
  weight: log2(m/u) on agreement, log2((1-m)/(1-u)) on disagreement.
* The composite score for a pair is the sum of its field weights, and
  pairs are classified as match / possible match / non-match against
  two cutoffs.

m/u probabilities can be supplied directly, or estimated from the
observed agreement patterns with an EM algorithm in the style described
by Winkler (1988), under the standard conditional-independence
assumption between fields.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

EPS = 1e-6


@dataclass
class FieldSpec:
    """One compared field.

    name: label used in reports.
    column: source dataframe column.
    method: 'jw' (Jaro-Winkler fuzzy), 'near' (Jaro-Winkler with a
        stricter threshold, for near-duplicate DOB/ID typos), or
        'exact' (normalized string equality).
    threshold: similarity cutoff for 'jw'/'near' to count as agreement.
    """

    name: str
    column: str
    method: str = "jw"
    threshold: float = 0.85


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1.0 - EPS)


class FellegiSunterModel:
    def __init__(self, field_names: List[str], m_prob: np.ndarray, u_prob: np.ndarray, prior: float = 0.05):
        if len(field_names) != len(m_prob) or len(field_names) != len(u_prob):
            raise ValueError("field_names, m_prob, u_prob must have the same length")
        self.field_names = list(field_names)
        self.m_prob = _clip(np.asarray(m_prob, dtype=float))
        self.u_prob = _clip(np.asarray(u_prob, dtype=float))
        self.prior = float(np.clip(prior, EPS, 1.0 - EPS))

    @classmethod
    def with_default_probabilities(cls, field_names: List[str], m_default: float = 0.9,
                                    u_default: float = 0.1, prior: float = 0.05) -> "FellegiSunterModel":
        n = len(field_names)
        return cls(field_names, np.full(n, m_default), np.full(n, u_default), prior)

    def agreement_weights(self) -> np.ndarray:
        """log2(m/u), the score contribution when a field agrees."""
        return np.log2(self.m_prob / self.u_prob)

    def disagreement_weights(self) -> np.ndarray:
        """log2((1-m)/(1-u)), the score contribution when a field disagrees."""
        return np.log2((1.0 - self.m_prob) / (1.0 - self.u_prob))

    def max_possible_score(self) -> float:
        """Composite score when every field agrees (used for auto thresholds)."""
        return float(np.sum(np.maximum(self.agreement_weights(), 0.0)))

    def min_possible_score(self) -> float:
        """Composite score when every field disagrees."""
        return float(np.sum(np.minimum(self.disagreement_weights(), 0.0)))

    def score_pairs(self, agreement_matrix: np.ndarray) -> np.ndarray:
        """Composite Fellegi-Sunter score for each row of a 0/1 agreement matrix."""
        agree_w = self.agreement_weights()
        disagree_w = self.disagreement_weights()
        return agreement_matrix @ agree_w + (1 - agreement_matrix) @ disagree_w

    def fit_em(self, agreement_matrix: np.ndarray, max_iter: int = 20, tol: float = 1e-6) -> dict:
        """Estimate m, u, and the match prior with expectation-maximization.

        Under the field-independence assumption, the observed agreement
        patterns are a two-component mixture (matches vs. non-matches).
        This is the standard Winkler-style EM fit for Fellegi-Sunter
        parameters when no labeled training pairs are available.

        Returns a dict with 'iterations' and 'log_likelihood' history.
        """
        A = np.asarray(agreement_matrix, dtype=float)
        n_pairs, n_fields = A.shape
        if n_fields != len(self.field_names):
            raise ValueError("agreement_matrix column count does not match field_names")
        if n_pairs == 0:
            return {"iterations": 0, "log_likelihood": []}

        m = self.m_prob.copy()
        u = self.u_prob.copy()
        p = self.prior

        ll_history = []
        prev_ll = None

        for iteration in range(1, max_iter + 1):
            # E-step: posterior P(match | pattern) per pair, via log-space
            # to avoid underflow with many fields.
            log_m = A @ np.log(m) + (1 - A) @ np.log(1.0 - m)
            log_u = A @ np.log(u) + (1 - A) @ np.log(1.0 - u)

            log_num_match = np.log(p) + log_m
            log_num_nonmatch = np.log(1.0 - p) + log_u

            # log-sum-exp for the denominator
            stacked = np.stack([log_num_match, log_num_nonmatch], axis=0)
            max_stacked = np.max(stacked, axis=0)
            log_denom = max_stacked + np.log(np.sum(np.exp(stacked - max_stacked), axis=0))

            g = np.exp(log_num_match - log_denom)  # posterior P(match) per pair
            g = _clip(g)

            log_likelihood = float(np.sum(log_denom))
            ll_history.append(log_likelihood)

            # M-step
            sum_g = np.sum(g)
            sum_1mg = np.sum(1.0 - g)
            new_m = (g @ A) / sum_g if sum_g > EPS else m
            new_u = ((1.0 - g) @ A) / sum_1mg if sum_1mg > EPS else u
            new_p = float(np.mean(g))

            m = _clip(new_m)
            u = _clip(new_u)
            p = float(np.clip(new_p, EPS, 1.0 - EPS))

            if prev_ll is not None and abs(log_likelihood - prev_ll) < tol:
                prev_ll = log_likelihood
                break
            prev_ll = log_likelihood

        self.m_prob = m
        self.u_prob = u
        self.prior = p

        return {"iterations": iteration, "log_likelihood": ll_history}

    @staticmethod
    def classify(scores: np.ndarray, upper: float, lower: float) -> List[str]:
        """Bucket composite scores into match / possible_match / non_match."""
        labels = []
        for s in scores:
            if s >= upper:
                labels.append("match")
            elif s >= lower:
                labels.append("possible_match")
            else:
                labels.append("non_match")
        return labels
