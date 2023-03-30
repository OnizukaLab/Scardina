"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.
"""
import os
import time
import re
import logging
from typing import Dict, List, Set, Tuple, Union
from functools import reduce
import operator

import numpy as np
import pandas as pd
import networkx as nx
import hypernetx as hx
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import more_itertools as mi

import scardina.models
import scardina.common
import scardina.util
from scardina.common import Predicate


def op_like(all_vals, x):
    #  NOTE: poor performance
    pattern = re.compile(re.escape(x).replace("_", ".").replace("%", ".*"))
    like_fn = np.vectorize(lambda val: re.fullmatch(pattern, val) is not None)
    return like_fn(all_vals)


def op_not_like(all_vals, x):
    #  NOTE: poor performance
    pattern = re.compile(re.escape(x).replace("_", ".").replace("%", ".*"))
    not_like_fn = np.vectorize(lambda val: re.fullmatch(pattern, val) is None)
    return not_like_fn(all_vals)


def op_in(all_vals, x):
    # np.isin alternative (np.isin is very slow)
    x_set = set(x)  # TODO: cache x_set
    return np.vectorize(lambda v: v in x_set)(all_vals)


"""
class _OPS:
    def __init__(self):
        self.__OPS = {
            ">": np.greater,
            "<": np.less,
            ">=": np.greater_equal,
            "<=": np.less_equal,
            "=": np.equal,
            "!=": lambda all_vals, x: ~np.equal(all_vals, x),
            "<>": lambda all_vals, x: ~np.equal(all_vals, x),
            "IN": op_in,
            "BETWEEN": lambda all_vals, lb_ub: np.logical_and(
                np.greater_equal(all_vals, lb_ub[0]), np.less_equal(all_vals, lb_ub[1])
            ),
            "IS": lambda all_vals, x: np.full_like(all_vals, False, dtype=np.bool)
            if x is None
            else np.equal(all_vals, x),
            "IS NOT": lambda all_vals, x: np.full_like(all_vals, True, dtype=np.bool)
            if x is None
            else ~np.equal(all_vals, x),
            "LIKE": op_like,
            "NOT LIKE": op_not_like,
            "ALL_TRUE": lambda all_vals, x: np.full_like(all_vals, True, dtype=np.bool),
            "ALL_FALSE": lambda all_vals, x: np.full_like(
                all_vals, False, dtype=np.bool
            ),
        }

    def __getitem__(self, key):
        return _OPS._wrap(self.__OPS[key], key)

    @staticmethod
    def _wrap(f, key):
        def _inner(all_vals, x):
            s = time.time()
            res = f(all_vals, x)
            print(f"{key} ({all_vals.shape}, {x}): {time.time() - s}sec")
            return res

        return _inner

OPS = _OPS()
"""

OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
    "!=": lambda all_vals, x: ~np.equal(all_vals, x),
    "<>": lambda all_vals, x: ~np.equal(all_vals, x),
    "IN": op_in,
    "BETWEEN": lambda all_vals, lb_ub: np.logical_and(
        np.greater_equal(all_vals, lb_ub[0]), np.less_equal(all_vals, lb_ub[1])
    ),
    "IS": lambda all_vals, x: np.full_like(all_vals, False, dtype=np.bool)
    if x is None
    else np.equal(all_vals, x),
    "IS NOT": lambda all_vals, x: np.full_like(all_vals, True, dtype=np.bool)
    if x is None
    else ~np.equal(all_vals, x),
    "LIKE": op_like,
    "NOT LIKE": op_not_like,
    "ALL_TRUE": lambda all_vals, x: np.full_like(all_vals, True, dtype=np.bool),
    "ALL_FALSE": lambda all_vals, x: np.full_like(all_vals, False, dtype=np.bool),
}

PROJECT_OPS = {
    "<": "<=",
    ">": ">=",
    "!=": "ALL_TRUE",
    "<=": "<=",
    ">=": ">=",
}
PROJECT_OPS_LAST = {
    "<": "<",
    ">": ">",
    "!=": "!=",
}
PROJECT_OPS_DOMINANT = {
    "<=": "<",
    ">=": ">",
    "<": "<",
    ">": ">",
    "!=": "!=",
}


def query_to_predicates(query):
    predicates = []
    for c, o, v in zip(query[0], query[1], query[2]):
        predicates.append(Predicate(c, o, v))
    for t in query[3]:
        predicates.append(Predicate(f"__in__:{t}", "=", True))
    return predicates


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self, log: logging.Logger):
        self.log: logging.Logger = log

        self.query_starts = []
        self.elapsed_time_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = "CardEst"

    def query(self, query):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    # @profile
    def _estimate(
        self,
        table: scardina.common.CsvTable,
        predicates: List[Predicate],
        sample_size: int,
        model: scardina.models.NAR,
        samples: torch.Tensor,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        *,
        fanout_cols: Dict[Tuple[str, str], scardina.common.Column] = {},
        inherited_samples: Dict[str, torch.Tensor] = {},
        common_cols: List[scardina.common.Column] = [],
    ):
        norm_preds = CardEst._normalize_predicates(predicates, table)
        fact_preds = self._project_predicates(norm_preds, table)

        conditioned_cols = CardEst._ordering(norm_preds, self.order, table)

        # 共通属性のうちまだサンプルされていないものをconditionedに追加
        for col in common_cols:
            if (
                col.name not in inherited_samples
                and col not in conditioned_cols
                and not col.is_fanout()
            ):
                c_tbl = re.sub(r"\..+$", "", col.name)
                c_tbl = re.sub(r"^.+:", "", c_tbl)
                tbl_size = self.db.schema.g.nodes[c_tbl]["size"]
                if self.skip_high_card_cols and col.dist_size > tbl_size / 2:
                    self.log.log(9, f"skip {col.name} for common sample")
                    continue
                conditioned_cols.append(col)

        if len(inherited_samples) > 0:
            for scol in table.scols:
                if scol.name in inherited_samples:
                    samples[:, scol.scol_idx] = inherited_samples[scol.name]
                    model.embs.encode(samples, scol.scol_idx, inputs)

        prob = torch.ones(
            sample_size, dtype=torch.float, device=self.device
        )  # to be aggregated
        for col in conditioned_cols:
            factor_masks = None
            for scol in col.scols[::-1]:
                has_mask = False

                dist = self._predict(model, logits, scol.scol_idx)

                preds = [p for p in fact_preds if p.c == scol.name]
                if len(preds) > 0:
                    # cols with predicates
                    # will be used to calc prob
                    valid, has_mask = CardEst._get_valids(
                        scol, preds, sample_size, factor_masks
                    )
                    assert isinstance(valid, float) or dist.shape[-1] == valid.shape[-1]
                    dist *= torch.as_tensor(valid, device=self.device)
                    curr_pred_prob = dist.sum(dim=1)

                    # dist was vanished by curr predicate
                    # use uniform sampling
                    vanished = (curr_pred_prob <= 0).view(-1, 1)
                    dist.masked_fill_(vanished, 1.0)

                    prob *= curr_pred_prob
                else:
                    # col without predicates
                    # just sample them
                    pass

                # if last scol, needs to mask dist to avoid overflow
                # TODO: last scol -> not first scol
                if scol.is_factorized and scol.scol_idx == col.scol_idxes[0]:
                    if len(col.scol_idxes[1:]) == 1:
                        reach_upper = (
                            samples[:, col.scol_idxes[1:]] >= col.fact_upper[1:]
                        ).squeeze()
                    else:
                        """
                        TODO: support #scols>2
                        dom = A B C D E
                        s0  = 0 1 0 1 0 ↑
                        s1  = 0 0 1 1 0 ↑
                        s2  = 0 0 0 0 1 ↑ est order
                        s3  = 0 0 0 0 0 ↑
                                 upper^
                        when s3=0, no restriction
                        when s3=0 & s2=0, no restriction
                        when s3=0 & s2=1, s1=0 & s0=0
                        ...
                        """
                        assert False, "FIXME"
                        prec_sample = None
                        upper = None
                        for sc in col.scols[1:]:
                            if prec_sample is None:
                                prec_sample = samples[:, sc.scol_idx].detach().clone()
                                upper = col.fact_upper[0].detach()
                            else:
                                prec_sample |= (
                                    samples[:, sc.scol_idx] - 1
                                ) << sc.fact_offset
                                upper |= (
                                    col.fact_upper[sc.fact_idx] - 1
                                ) << sc.fact_offset
                        reach_upper = prec_sample > upper
                    last_fact_upper = col.fact_upper[0]
                    if reach_upper.sum() > 0 and len(preds) > 0:
                        # TODO: reflect this masking to prob calculation (zero-filling overed range and normalizing)
                        self.log.warning(
                            f"{col.name} maybe overestimate ({reach_upper.sum().item()} samples): {dist[reach_upper, last_fact_upper:].sum(dim=1)}"
                        )
                    dist[reach_upper, last_fact_upper:] = 0

                samples[:, scol.scol_idx] = torch.multinomial(
                    dist, num_samples=1, replacement=True
                ).squeeze(1)

                # last scol (descending order) doesn't have to update factor masks
                if has_mask and scol.scol_idx != col.scol_idxes[0]:
                    factor_masks = CardEst._update_factor_masks(
                        samples[:, scol.scol_idx].view(-1),
                        preds,
                        factor_masks,
                    )

                model.embs.encode(samples, scol.scol_idx, inputs)
                logits = model.forward_w_encoded(inputs)

            # DEBUG
            if self.log.level <= 10:
                # padding for MASK and NULL
                distinct_vals = np.concatenate([[None, None], col.distinct_vals])

                preds = [p for p in norm_preds if p.c == col.name]
                has_preds = len(preds) > 0
                if col.is_factorized:
                    reconstructed_samples = torch.zeros_like(samples[:, 0])
                    for scol in col.scols:
                        reconstructed_samples |= (
                            samples[:, scol.scol_idx] - 1
                        ) << scol.fact_offset
                    reconstructed_samples = reconstructed_samples.cpu().numpy()
                    over = (reconstructed_samples - 2) >= len(distinct_vals)
                    if over.sum() > 0:
                        self.log.warning(
                            f"OOD{'@' if has_preds else ''} ({col.name}({len(col.scols)})): {over.sum()}, {[(p.o, p.v) for p in preds if p.c == col.name]}"
                        )
                        self.log.debug(reconstructed_samples[over])
                        reconstructed_samples[over] = len(distinct_vals) - 1
                else:
                    reconstructed_samples = samples[:, col.scol_idxes[0]].cpu().numpy()

                # sample_values = distinct_vals[reconstructed_samples]
                # self.log.debug(f"{col.name}: {sample_values[:10]}")
                if has_preds:
                    try:
                        domain = None
                        for pred in preds:
                            dom_per_pred = np.append(
                                [
                                    False,
                                    True
                                    if (pred.o == "IS" and pred.v is None)
                                    or (pred.o == "IS NOT" and pred.v is not None)
                                    else False,
                                ],
                                OPS[pred.o](col.distinct_vals, pred.v),
                            )

                            if domain is None:
                                domain = dom_per_pred
                            else:
                                domain &= dom_per_pred

                        x = domain[reconstructed_samples]
                    except Exception as e:
                        self.log.error(e)
                        self.log.error(domain.shape)
                        self.log.error(len(distinct_vals))
                        self.log.error(reconstructed_samples.max())
                        self.log.error(reconstructed_samples)
                        self.log.error(over)
                    if len(x) - x.sum() > 0:
                        self.log.warning(
                            f"NOTMATCHED ({col.name}({len(col.scols)})): {len(x)-x.sum()}, {[(p.o, p.v) for p in preds if p.c == col.name]}"
                        )

        fanouts = {}
        for from_to, col in fanout_cols.items():
            assert not col.is_factorized, "no impl for factored fanout cols"
            assert len(col.scol_idxes) == 1
            scol_idx = col.scol_idxes[0]
            logits = model.forward_w_encoded(inputs)
            if model.embs.dom_sizes[scol_idx] == 1:  # cont fanout
                log2_fo = model.embs.decode_as_raw_val(logits, scol_idx)[:, 1]
                fo = 2**log2_fo
                fanouts[from_to] = torch.maximum(fo, torch.ones_like(fo))
                samples[:, scol_idx] = fanouts[from_to]
            else:
                fo_dist = torch.softmax(model.embs.decode_as_logit(logits, scol_idx), 1)
                fo_dist[:, :2] = 0  # fo is neither mask nor null
                fo = torch.multinomial(fo_dist, num_samples=1, replacement=True)
                fanouts[from_to] = table.scols[scol_idx].to_val[fo].squeeze(1)
                assert fo.min() >= 2
                assert fanouts[from_to].min() >= 1
                samples[:, scol_idx] = fo.squeeze(1)
            model.embs.encode(samples, scol_idx, inputs)

        if inherited_samples is not None:
            common_samples = inherited_samples
            for col in common_cols:
                for scol in col.scols:
                    common_samples[scol.name] = samples[:, scol.scol_idx]
            return prob, fanouts, common_samples
        else:
            return prob, fanouts

    def _predict(
        self,
        model: scardina.models.NAR,
        logits: torch.Tensor,
        scol_idx: int,
    ):
        dist = torch.softmax(model.embs.decode_as_logit(logits, scol_idx), 1)

        # mask never appears
        dist[:, 0] = 0

        return dist

    def on_start(self):
        self.query_starts.append(time.time())

    def on_end(self):
        elapsed_time_ms = (time.time() - self.query_starts[-1]) * 1e3
        self.elapsed_time_ms.append(elapsed_time_ms)
        return elapsed_time_ms

    def add_err(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts,
            self.elapsed_time_ms,
            self.errs,
            self.est_cards,
            self.true_cards,
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.elapsed_time_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(
            est.name,
            "max",
            np.max(est.errs),
            "99th",
            np.quantile(est.errs, 0.99),
            "95th",
            np.quantile(est.errs, 0.95),
            "median",
            np.quantile(est.errs, 0.5),
            "time_ms",
            np.mean(est.elapsed_time_ms),
        )

    @staticmethod
    def _extract_matched_predicates_by_tables(
        predicates: List[Predicate], used_table_names: List[str]
    ):
        matched = []
        rest = []
        for pred in predicates:
            found = False
            for t in used_table_names:
                if pred.c.startswith(f"{t}.") or pred.c == f"__in__:{t}":
                    matched.append(pred)
                    found = True
                    break
            if not found or pred.c.startswith("__in__:"):
                # always keeps existence markers
                rest.append(pred)
        return matched, rest

    @staticmethod
    def _normalize_predicates(
        predicates: List[Predicate], table: scardina.common.CsvTable
    ) -> List[Predicate]:
        ops_to_be_in = {"IN", "NOT IN", "LIKE", "NOT LIKE"}
        preds_each_col: Dict[str, List[Predicate]] = {}
        for pred in predicates:
            if pred.c not in preds_each_col:
                preds_each_col[pred.c] = []
            preds_each_col[pred.c].append(pred)

        norm_preds: List[Predicate] = []
        for c, preds in preds_each_col.items():
            num_ops_to_be_in = len([1 for p in preds if p.o in ops_to_be_in])
            num_null_checks = len(
                [
                    1
                    for p in preds
                    if (p.o == "IS" and p.v is None)
                    or (p.o == "IS NOT" and p.v is None)
                ]
            )
            assert (
                len(preds) == num_ops_to_be_in + num_null_checks
                or num_ops_to_be_in == 0
            ), f"（未実装のため，）col {c} を対象とした全てのOPがINに変換されるものか，INに変換されるOPが存在しない必要がある（NULL判定とは共存可能）"

            if len(preds) > 1 and all([p.o in ops_to_be_in for p in preds]):
                # Merge OPs which will be converted into IN for each col
                col = table.cols[table.col_name_to_idx(c)]
                valid = np.ones_like(col.distinct_vals, dtype=bool)
                for pred in preds:
                    valid &= OPS[pred.o](col.distinct_vals, pred.v)
                norm_preds.append(
                    Predicate(c, "IN", col.distinct_vals[valid], no_ood=True)
                )
            else:
                for pred in preds:
                    col = table.cols[table.col_name_to_idx(pred.c)]
                    if pred.o in {"ALL_TRUE", "ALL_FALSE"}:
                        norm_preds.append(pred)
                    elif pred.o in {"LIKE", "NOT LIKE"}:
                        valid = OPS[pred.o](col.distinct_vals, pred.v)
                        norm_preds.append(
                            Predicate(
                                pred.c, "IN", col.distinct_vals[valid], no_ood=True
                            )
                        )
                    elif pred.o == "BETWEEN":
                        norm_preds.append(Predicate(pred.c, ">=", pred.v[0]))
                        norm_preds.append(Predicate(pred.c, "<=", pred.v[1]))
                    elif pred.v is None:
                        if pred.o == "IS":
                            norm_preds.append(
                                Predicate(pred.c, pred.o, pred.v, no_ood=True)
                            )
                        elif pred.o == "IS NOT":
                            norm_preds.append(
                                Predicate(pred.c, pred.o, pred.v, no_ood=True)
                            )
                        elif pred.o == "=":
                            norm_preds.append(
                                Predicate(pred.c, "IS", pred.v, no_ood=True)
                            )
                        elif pred.o in {"!=", "<>"}:
                            norm_preds.append(
                                Predicate(pred.c, "IS NOT", pred.v, no_ood=True)
                            )
                    elif pred.o == "<>":
                        norm_preds.append(Predicate(pred.c, "!=", pred.v))
                    else:
                        norm_preds.append(pred)

        return norm_preds

    @staticmethod
    def _project_predicates(
        predicates: List[Predicate], table: scardina.common.CsvTable
    ) -> List[Predicate]:
        allowed = {"=", "!=", "<", ">", "<=", ">=", "IN", "IS", "IS NOT"}
        for predicate in predicates:
            assert predicate.o in allowed

        fact_preds: List[Predicate] = []
        for pred in predicates:
            col = table.cols[table.col_name_to_idx(pred.c)]

            if not col.is_factorized:
                # This column is not factorized.
                fact_preds.append(pred)  # as-is
            else:
                # project matched values into indices on col
                if pred.o == "IN" and len(pred.v) > 0:
                    cate_ids = col.discretize_values(pred.v, drop_ood=(not pred.no_ood))
                elif (
                    pred.o != "IN"
                    and pred.v is not None
                    and pred.v in col.distinct_vals
                ):
                    cate_ids = col.discretize_values(pred.v)  # assume no ood values

                for scol in col.scols:
                    # This column is factorized.  For IN queries, we need to
                    # map the projection overall elements in the tuple.
                    if pred.o == "IN":
                        if len(pred.v) == 0:
                            fact_preds.append(
                                Predicate(scol.name, "ALL_FALSE", None, no_ood=True)
                            )
                        else:
                            # project indices on col into indices on scol
                            scate_ids = scol.project_values(cate_ids)

                            fact_preds.append(
                                Predicate(
                                    scol.name, pred.o, scate_ids, "IN", no_ood=True
                                )
                            )
                    # IS_NULL/IS_NOT_NULL Handling.
                    # IS_NULL+column has null value -> convert to = 0.
                    # IS_NULL+column has no null value -> return False for
                    #   everything.
                    # IS_NOT_NULL+column has null value -> convert to > 0.
                    # IS_NOT_NULL+column has no null value -> return True for
                    #   everything.
                    elif pred.v is None:
                        if pred.o == "IS":
                            fact_preds.append(
                                Predicate(
                                    scol.name,
                                    "=",
                                    scol.project_values(1),
                                    "=",
                                    no_ood=True,
                                )
                            )
                        elif pred.o == "IS NOT":
                            fact_preds.append(
                                Predicate(
                                    scol.name,
                                    "!=",
                                    scol.project_values(1),
                                    "!=",
                                    no_ood=True,
                                )
                            )
                        else:
                            assert False, "Operator {} not supported".format(pred.o)
                    else:
                        # Handling =/<=/>=/</>.
                        # If the original column has a NaN, then we shoudn't
                        # include this in the result.  We can ensure this by
                        # adding a >0 predicate on the fact col.  Only need to
                        # do this if the original predicate is <, <=, or !=.
                        if pred.o in ["<=", "<", "!="] and np.any(
                            pd.isnull(col.distinct_vals)
                        ):
                            fact_preds.append(
                                Predicate(
                                    scol.name,
                                    "!=",
                                    scol.project_values(1),
                                    "!=",
                                    no_ood=True,
                                )
                            )
                        if pred.v not in col.distinct_vals:
                            # Handle cases where value is not in the column
                            # vocabulary.
                            assert pred.o in ["=", "!=", "<>"]
                            if pred.o == "=":
                                # Everything should be False.
                                fact_preds.append(
                                    Predicate(scol.name, "ALL_FALSE", None, no_ood=True)
                                )
                            elif pred.o == "!=":
                                # Everything should be True.
                                # Note that >0 has already been added,
                                # so there are no NULL results.
                                fact_preds.append(
                                    Predicate(scol.name, "ALL_TRUE", None, no_ood=True)
                                )
                        else:
                            assert isinstance(pred.v, str) or not hasattr(
                                pred.v, "__len__"
                            )
                            assert (
                                len(cate_ids) == 1
                            ), "value with =,!=,<,>,<=,>= should be single, in-domain, and non-null. out-of-domain values should be given special treatment."
                            scate_id = scol.project_values(cate_ids)
                            o = PROJECT_OPS.get(pred.o, pred.o)
                            dom_o = PROJECT_OPS_DOMINANT.get(pred.o, pred.o)
                            if dom_o not in PROJECT_OPS_DOMINANT.values():
                                dom_o = None
                            fact_preds.append(Predicate(scol.name, o, scate_id, dom_o))
        return fact_preds

    @staticmethod
    def _get_valids(
        scol: scardina.common.Column,
        predicates: List[Predicate],
        num_samples: int,
        factor_masks: List[np.ndarray] = None,
    ):
        """Returns a valid mask of shape (), (size(col)) or (N, size(col)).

        For columns that are not factorized, the first dimension is trivial.
        For columns that are not filtered, both dimensions are trivial.
        """
        # Indicates whether valid values for this column depends on samples
        # from previous columns.  Only used for factorized columns with
        # >/</>=/<=/!=/IN.By default this is False.
        has_mask = False
        # Column i.
        if len(predicates) > 0:
            # There exists a filter.
            if not scol.is_scol:
                # This column is not factorized.
                valids = []
                for pred in predicates:
                    valid = OPS[pred.o](scol.distinct_vals, pred.v).astype(
                        np.float32, copy=False
                    )
                    valid = np.append(
                        [
                            0.0,
                            # ^ complement mask token
                            1.0
                            if (pred.o == "IS" and pred.v is None)
                            or (pred.o == "IS NOT" and pred.v is not None)
                            else 0.0,
                        ],
                        valid,
                    )
                    valids.append(valid)

                valid = np.logical_and.reduce(valids, 0).astype(np.float32, copy=False)
            else:
                # This column is factorized.  `valid` stores the valid values
                # for this column for each operator.  At the very end, combine
                # the valid values for the operatorsvia logical and.
                valid = np.ones((len(predicates), len(scol.distinct_vals)), np.bool)
                # for i, (o, v) in enumerate(zip(op, val)):
                for i, pred in enumerate(predicates):
                    # Handle the various operators.  For ops with a mask, we
                    # add a new dimension so that we can add the mask.  Refer
                    # to `update_factor_mask` for description
                    # of self.factor_mask.
                    if (
                        pred.o in PROJECT_OPS.values()
                        or pred.o in PROJECT_OPS_LAST.values()
                    ):
                        valid[i] &= OPS[pred.o](scol.distinct_vals, pred.v)
                        has_mask = True
                        if scol.fact_idx < len(scol.orig_col.scols) - 1:
                            if len(valid.shape) != 3:
                                valid = np.tile(
                                    np.expand_dims(valid, 1), (1, num_samples, 1)
                                )
                            assert valid.shape == (
                                len(predicates),
                                num_samples,
                                len(scol.distinct_vals),
                            )
                            assert factor_masks is not None, "here"
                            expanded_mask = np.expand_dims(factor_masks[i], 1)
                            assert expanded_mask.shape == (num_samples, 1)
                            valid[i] |= expanded_mask
                    # IN is special case.
                    elif pred.o == "IN":
                        has_mask = True
                        matches = scol.distinct_vals[:, None] == pred.v
                        assert matches.shape == (
                            len(scol.distinct_vals),
                            len(pred.v),
                        ), matches.shape
                        if scol.fact_idx < len(scol.orig_col.scols) - 1:
                            assert factor_masks[i] is not None
                            if len(valid.shape) != 3:
                                valid = np.tile(
                                    np.expand_dims(valid, 1), (1, num_samples, 1)
                                )
                            assert valid.shape == (
                                len(predicates),
                                num_samples,
                                len(scol.distinct_vals),
                            ), valid.shape
                            matches = np.tile(matches, (num_samples, 1, 1))
                            expanded_mask = np.expand_dims(factor_masks[i], 1)
                            matches &= expanded_mask
                        valid[i] = np.logical_or.reduce(matches, axis=-1).astype(
                            np.float32, copy=False
                        )
                    else:
                        valid[i] &= OPS[pred.o](scol.distinct_vals, pred.v)
                valid = np.logical_and.reduce(valid, 0).astype(np.float32, copy=False)
                assert valid.shape == (
                    num_samples,
                    len(scol.distinct_vals),
                ) or valid.shape == (len(scol.distinct_vals),), valid.shape

                # complement MASK
                if valid.shape == (len(scol.distinct_vals),):
                    # for head fact col
                    valid = np.append([0.0], valid)
                elif valid.shape == (num_samples, len(scol.distinct_vals)):
                    # for tail fact cols
                    valid = np.concatenate([np.zeros((num_samples, 1)), valid], 1)
        else:
            # This column is unqueried.  All values are valid.
            valid = 1.0

        assert (
            isinstance(valid, float)
            or valid.shape == (scol.dist_size,)
            or valid.shape == (num_samples, scol.dist_size)
        )
        return valid, has_mask

    @staticmethod
    def _update_factor_masks(
        samples: torch.Tensor,
        predicates: List[Predicate],
        factor_masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        samples = samples.cpu().numpy()
        if factor_masks is None:
            factor_masks = [None] * len(predicates)
        for i, pred in enumerate(predicates):
            if pred.dom_o == "IN":
                # Mask for IN should be size (N, len(v))
                new_mask = samples[:, None] == pred.v
                # In the example above, for each sample s_i,
                # new_mask stores
                # [
                #   (s_i_1 == y1 and s_i_2 == y2 ...),
                #   (s_i_1 == z1 and s_i_2 == z2 ...),
                #   ...
                # ]
                # As we sample, we &= the current mask with previous masks.
                assert new_mask.shape == (len(samples), len(pred.v)), new_mask.shape
                if factor_masks[i] is not None:
                    new_mask &= factor_masks[i]
            elif pred.dom_o in PROJECT_OPS_DOMINANT.values():
                new_mask = OPS[pred.dom_o](samples, pred.v)
                if factor_masks[i] is not None:
                    new_mask |= factor_masks[i]
            else:
                assert (
                    pred.dom_o is None
                ), "This dominant operator ({}) is not supported.".format(pred.dom_o)
                new_mask = np.zeros_like(samples, dtype=np.bool)
                if factor_masks[i] is not None:
                    new_mask |= factor_masks[i]
            factor_masks[i] = new_mask
        return factor_masks

    @staticmethod
    def _ordering(
        predicates: List[Predicate],
        order: str,
        table: scardina.common.CsvTable,
    ) -> List[scardina.common.Column]:
        cols = {}
        for pred in predicates:
            col = table.cols[table.col_name_to_idx(pred.c)]
            if col.name not in cols:
                if order in {"prop", "prop-inv", "prop-ratio", "domain-size"}:
                    # roughly ignore MASK and NULL
                    n_match = OPS[pred.o](col.distinct_vals, pred.v).sum()
                else:
                    n_match = 0
                cols[col] = n_match

        if order == "prop":
            ordered_col_idxes = list(
                dict(
                    sorted(
                        cols.items(),
                        key=lambda kv: kv[1],
                    )
                ).keys()
            )  # sort by the num of el satisfying predicates (proposal)
        elif order == "prop-inv":
            ordered_col_idxes = list(
                dict(
                    sorted(
                        cols.items(),
                        key=lambda kv: kv[1],
                    )
                ).keys()
            )
            ordered_col_idxes = list(reversed(ordered_col_idxes))  # propinv (for eval)
        elif order == "prop-ratio":
            ordered_col_idxes = list(
                dict(
                    sorted(
                        cols.items(),
                        key=lambda kv: kv[1] / kv[0].dist_size,
                    )
                ).keys()
            )
        elif order == "domain-size":
            ordered_col_idxes = list(
                dict(
                    sorted(
                        cols.items(),
                        key=lambda kv: kv[0].dist_size,
                    )
                ).keys()
            )
        elif order == "nat":
            ordered_col_idxes = list(
                dict(
                    sorted(
                        cols.items(),
                        key=lambda kv: kv[0].col_idx,
                    )
                ).keys()
            )
        elif order == "inv":
            ordered_col_idxes = list(
                dict(
                    sorted(
                        cols.items(),
                        key=lambda kv: kv[0].col_idx,
                    )
                ).keys()
            )
            ordered_col_idxes = list(
                reversed(sorted(ordered_col_idxes))
            )  # inv order (for eval)
        else:
            assert False
        return ordered_col_idxes


class ProgressiveSamplingUR(CardEst):
    def __init__(
        self,
        model: pl.LightningModule,
        db: scardina.common.DB,
        params_dict: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
        log: logging.Logger,
        *,
        dump_intermediates: bool = False,
    ):
        super().__init__(log)

        # FIXME, TMP: not work with single table
        if db.vtable is None:
            db.vtable = db.tables[db.name]

        assert (
            model.n_cols == db.vtable.n_scols
        ), f"{model.n_cols} != {db.vtable.n_scols}"

        params = list(params_dict.values())[0]

        self.model = model
        self.db = db

        self.name = "ProgressiveSamplingNARUR"
        self.sample_size = params["static"]["eval_sample_size"]
        self.order = params["static"]["eval_order"]
        self.n_full_rows = self.db.vtable.n_rows
        self.device = params["static"]["device"]
        self.dump_intermediates = dump_intermediates
        self.skip_high_card_cols = params["static"]["eval_skip_high_card_cols"]

        with torch.inference_mode():
            n_scols = self.db.vtable.n_scols
            dtype = torch.float if self.model.embs.cont_fanout else torch.long
            inputs = torch.empty((1, self.model.embs.sum_of_dims), device=self.device)
            samples = torch.zeros(
                (1, n_scols),
                dtype=dtype,
                device=self.device,
                requires_grad=False,
            )
            self.model.embs.encode_all(
                samples,
                inputs,
            )
            self.init_samples = samples
            self.init_inputs = inputs
            self.init_logits = self.model.forward_w_encoded(inputs)

    def __str__(self):
        return f"NAR_UR_{self.sample_size}"

    def _find_fanout_cols(
        self, used_table_names: Set[str]
    ) -> Dict[Tuple[str, str], scardina.common.Column]:
        if len(used_table_names) == len(self.db.tables):
            return {}

        table = self.db.vtable

        not_used_path_roots = []
        for used in used_table_names:
            for ngh in self.db.schema.g.neighbors(used):
                if ngh not in used_table_names:
                    not_used_path_roots.append((used, ngh))

        fanout_cols = {}
        for used, ngh in not_used_path_roots:
            forest = self.db.schema.g.copy().to_undirected()

            # bridge
            e = forest.edges[(used, ngh, 0)]
            col_name = f"{used}.__fanout__:{e[used]}={ngh}.{e[ngh]}"
            fanout_cols[(used, ngh)] = table.cols[table.col_name_to_idx(col_name)]

            # remove bridge for extracting subtree whose nodes (tables) are not used in query
            forest.remove_edge(used, ngh)

            # edges in subtree
            for u, v in nx.bfs_edges(forest, ngh):
                e = forest.edges[(u, v, 0)]
                col_name = f"{u}.__fanout__:{e[u]}={v}.{e[v]}"
                fanout_cols[(u, v)] = table.cols[table.col_name_to_idx(col_name)]

        return fanout_cols

    def query(self, query):
        used_table_names = query[3]
        predicates = query_to_predicates(query)

        with torch.inference_mode():
            self.on_start()

            fanout_cols = self._find_fanout_cols(set(used_table_names))
            sels, fanouts = self._estimate(
                self.db.vtable,
                predicates,
                self.sample_size,
                self.model,
                self.init_samples.repeat((self.sample_size, 1)),
                self.init_inputs.repeat((self.sample_size, 1)),
                self.init_logits.repeat((self.sample_size, 1)),
                fanout_cols=fanout_cols,
            )
            self.log.debug(
                f"P({self.db.vtable.name}) (≈{sels.min().item():.3f},{sels.median().item():.3f},{sels.max().item():.3f})"
            )
            for ts, fo in fanouts.items():
                sels /= fo
                self.log.debug(
                    f"/ F({ts[0]}->{ts[1]}) (≈{fo.min().item():.3f},{fo.median().item():.3f},{fo.max().item():.3f})"
                )

            sel = sels.mean().item()
            card = sel * self.n_full_rows
            self.log.debug(f"* |{self.db.vtable.name}| (={self.n_full_rows})")

            elapsed_time_ms = self.on_end()

            return (
                np.ceil(card).astype(dtype=np.int, copy=False).item(),
                elapsed_time_ms,
            )


class ProgressiveSamplingCIN(CardEst):
    def __init__(
        self,
        models: Dict[str, pl.LightningModule],
        db: scardina.common.DB,
        params_dict: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
        log: logging.Logger,
    ):
        super().__init__(log)
        assert isinstance(models, dict)
        assert db.vtable is None

        # Use randomly picked table's params (expecting the same params)
        # Note: In the future, we can specify individual params for each joined table
        params = list(params_dict.values())[0]

        self.model = models
        self.db = db

        self.name = "ProgressiveSamplingNARPJ"
        self.sample_size = params["static"]["eval_sample_size"]
        self.order = params["static"]["eval_order"]
        self.device = params["static"]["device"]
        self.skip_high_card_cols = params["static"]["eval_skip_high_card_cols"]

        with torch.inference_mode():
            self.init_samples = {}  # all samples are mask tokens
            self.init_inputs = {}  # all inputs are encoded mask tokens
            self.init_logits = {}
            for table_name, m in self.model.items():
                n_scols = self.db.tables[table_name].n_scols
                dtype = torch.float if m.embs.cont_fanout else torch.long
                inputs = torch.empty((1, m.embs.sum_of_dims), device=self.device)
                samples = torch.zeros(
                    (1, n_scols),
                    dtype=dtype,
                    device=self.device,
                    requires_grad=False,
                )
                m.embs.encode_all(
                    samples,
                    inputs,
                )
                self.init_samples[table_name] = samples
                self.init_inputs[table_name] = inputs
                self.init_logits[table_name] = m.forward_w_encoded(inputs)

    def __str__(self):
        return f"NAR_PJ_{self.sample_size}"

    @staticmethod
    def _check_join_edge(
        graph: nx.DiGraph, u: Tuple[str, str], v: Tuple[str, str]
    ) -> Tuple[bool, Tuple[Tuple[str, str], Tuple[str, str]]]:
        u_t = None
        u_c = None
        v_t = None
        v_c = None
        if (u[0], v[0]) in graph.edges:
            u_t, u_c = u
            v_t, v_c = v
        if (v[0], u[0]) in graph.edges:
            u_t, u_c = v
            v_t, v_c = u
        if u_t is None:
            return False, None

        edge = graph.edges[(u_t, v_t)]
        found = edge[u_t] == u_c and edge[v_t] == v_c
        join = ((u_t, u_c), (v_t, v_c)) if found else None
        return found, join

    @staticmethod
    def _check_join_edge_multi(
        graph: nx.MultiDiGraph, u: Tuple[str, str], v: Tuple[str, str]
    ) -> Tuple[bool, Tuple[Tuple[str, str], Tuple[str, str]]]:
        u_t = None
        u_c = None
        v_t = None
        v_c = None
        if (u[0], v[0]) in graph.edges:
            u_t, u_c = u
            v_t, v_c = v
        if (v[0], u[0]) in graph.edges:
            u_t, u_c = v
            v_t, v_c = u
        if u_t is None:
            return False, None

        for _, d in graph.get_edge_data(u_t, v_t).items():
            if d[u_t] == u_c and d[v_t] == v_c:
                join = ((u_t, u_c), (v_t, v_c))
                return True, join
        return False, None

    def _estimate_over_models(self, sample_size: int, query):
        joins = set()
        for ux, vx in query[4]:
            u = ux[0]
            v = vx[0]
            col_names = self.db.schema.g.get_edge_data(u, v)
            if col_names is None:
                col_names = self.db.schema.g.get_edge_data(v, u)

            found_direct_path = False
            if col_names is not None:
                for col_name in col_names.values():
                    if col_name[u] == ux[1] and col_name[v] == vx[1]:
                        joins.add(((u, col_name[u]), (v, col_name[v])))
                        found_direct_path = True
                        break
            if col_names is None or not found_direct_path:
                # When not found in direct path (no edges or no proper keys),
                # try translation FK-FK into FK-(some tables)-FK (e.g., FK-PK PK-FK, FK-PK PK-).
                debug_msg = f"Join translation: ({u}.{ux[1]}--{v}.{vx[1]}) into"
                found = 0
                # try on all simple paths (from shortest to longest)
                path_cands = nx.all_simple_paths(self.db.schema.g.to_undirected(), u, v)
                for path_u_to_v in sorted(
                    list(set(tuple(p) for p in path_cands)), key=lambda p: len(p)
                ):
                    for x, y in mi.windowed(path_u_to_v, 2):
                        col_names = self.db.schema.g.get_edge_data(x, y)
                        if col_names is None:
                            col_names = self.db.schema.g.get_edge_data(y, x)
                        for col_name in col_names.values():
                            if (x == u and col_name[x] == ux[1]) or (
                                y == v and col_name[y] == vx[1]
                            ):
                                joins.add(((x, col_name[x]), (y, col_name[y])))
                                found += 1
                                debug_msg += f" ({x}.{col_name[x]}--{y}.{col_name[y]})"

                                # add indirectly joined tables
                                query[3].add(x)
                                query[3].add(y)
                                break
                    if found == len(path_u_to_v) - 1:
                        break  # connected u and v w/ given keys
                self.log.debug(debug_msg)
        query_graph = nx.DiGraph()
        for join in joins:
            found, d_join = self._check_join_edge_multi(
                self.db.schema.g, join[0], join[1]
            )
            assert found, f"queried unexpected join: {join}"
            query_graph.add_edge(
                d_join[0][0],
                d_join[1][0],
                **{d_join[0][0]: d_join[0][1], d_join[1][0]: d_join[1][1]},
            )
        assert nx.is_tree(query_graph), "not tree query"

        # after join translation (or add existence marker manually)
        predicates = query_to_predicates(query)

        nodes_to_be_removed = set(self.db.schema.subschema_hg.nodes) - set(query[3])
        hedges_to_be_removed = set()
        edges_to_be_removed = set()
        if query_graph.number_of_edges() > 0:  # has joins
            for joined_table_name, subjoin_graph in self.db.schema.subschemas.items():
                used = False
                inner_tables = set(subjoin_graph.nodes)
                for x_t, y_t, d in subjoin_graph.edges(data=True):
                    found, _ = self._check_join_edge(
                        query_graph, (x_t, d[x_t]), (y_t, d[y_t])
                    )
                    if found:
                        used = True
                        inner_tables.discard(x_t)
                        inner_tables.discard(y_t)
                if used:
                    for not_used_table in inner_tables:
                        edges_to_be_removed.add((joined_table_name, not_used_table))
                else:
                    hedges_to_be_removed.add(joined_table_name)
        else:
            # single table query uses random (first) joined table
            # Note: smaller model may be better
            assert len(query[3]) == 1
            for joined_table_name, subjoin_graph in sorted(
                self.db.schema.subschemas.items(), key=lambda x: len(x[1].nodes)
            ):
                if list(query[3])[0] in subjoin_graph.nodes:
                    hedges_to_be_removed = self.db.schema.subschemas.keys() - set(
                        [joined_table_name]
                    )
                    break
        exact_subjoin_forest_bi = self.db.schema.subschema_hg.bipartite()
        exact_subjoin_forest_bi.remove_nodes_from(nodes_to_be_removed)
        exact_subjoin_forest_bi.remove_nodes_from(hedges_to_be_removed)
        exact_subjoin_forest_bi.remove_edges_from(edges_to_be_removed)
        exact_subjoin_forest_hg = hx.Hypergraph.from_bipartite(exact_subjoin_forest_bi)

        if self.log.level <= 10:
            plt.clf()
            layout = nx.circular_layout(self.db.schema.g)

            scardina.util.draw_schema_graph(self.db.schema.g, layout)
            hx.draw(
                self.db.schema.subschema_hg,
                pos=layout,
                with_node_labels=False,
                nodes_kwargs={"sizes": 0},
            )
            plt.savefig(os.path.join("out", "output_aj_schema.pdf"))
            plt.savefig(os.path.join("out", "output_aj_schema.svg"))
            plt.clf()

            # global schema
            scardina.util.draw_schema_graph(self.db.schema.g, layout)
            plt.savefig(os.path.join("out", "output_glb_schema.pdf"))
            plt.savefig(os.path.join("out", "output_glb_schema.svg"))

            # query graph: always a subgraph of self.db.schema.g
            # TODO: properly render parallel edges
            nx.draw(
                query_graph,
                pos=layout,
                width=8,
                alpha=0.5,
                edge_color="tab:orange",  # edge
                node_color="tab:orange",  # node surface
            )
            plt.savefig(os.path.join("out", "output_query_graph.pdf"))
            plt.savefig(os.path.join("out", "output_query_graph.svg"))

            # adjacent-joined schemas
            subschema_bi = self.db.schema.subschema_hg.bipartite()
            subschema_bi.remove_nodes_from(
                set(self.db.schema.subschema_hg.edges)
                - set(exact_subjoin_forest_hg.edges)
            )
            subjoin_forest_hg = hx.Hypergraph.from_bipartite(subschema_bi)
            hx.draw(
                subjoin_forest_hg,
                pos=layout,
                with_node_labels=False,
                nodes_kwargs={"sizes": 0},
            )
            plt.savefig(os.path.join("out", "output.pdf"))
            plt.savefig(os.path.join("out", "output.svg"))

        root_joined_table_name = max(
            exact_subjoin_forest_hg.edges,
            key=lambda edge: exact_subjoin_forest_hg.edges[edge].size(),
        )

        def build_tree_from_hg(v, visited, tree=nx.DiGraph()):
            tree.add_node(v)
            nghs = exact_subjoin_forest_hg.edge_neighbors(v)
            nghs = set(nghs) - visited
            visited |= nghs
            for ngh in nghs:
                tree.add_edge(v, ngh)
                subtree = build_tree_from_hg(ngh, visited)
                if len(subtree) > 0:
                    tree.add_edges_from(subtree.edges)
            return tree

        joined_table_tree = build_tree_from_hg(
            root_joined_table_name,
            set({root_joined_table_name}),
        )

        # cardinality root
        cards = self.db.tables[root_joined_table_name].n_rows
        self.log.debug(f"|{root_joined_table_name}| (={cards})")

        common_samples = {}
        bfs_tables = [root_joined_table_name] + [
            v for _, v in nx.bfs_edges(joined_table_tree, root_joined_table_name)
        ]
        for u in bfs_tables:
            vs = list(nx.neighbors(joined_table_tree, u))
            table = self.db.tables[u]
            used_table_names = set(exact_subjoin_forest_hg.edges[u])

            # If there are multiple common tables, remove extra ones
            # e.g.,   [{t5, t7, t9}, {t5, t6}, {t7, t8, t10}]
            # common w/ ^ j1          ^ j2      ^ j3
            # We can remove t6, t8, t9, and t10,
            # since some common tables remain even if remove them from all common parts
            #         [{t5, t7}, {t5}, {t7}]
            cand_subsequent_table_names = {}
            for v in vs:
                commons = (
                    exact_subjoin_forest_hg.edges[u].uidset
                    & exact_subjoin_forest_hg.edges[v].uidset
                )
                subsequents = (
                    exact_subjoin_forest_hg.edges[v].uidset
                    - exact_subjoin_forest_hg.edges[u].uidset
                )
                assert len(subsequents) > 0
                cand_subsequent_table_names[commons] = (
                    cand_subsequent_table_names.get(commons, set()) | subsequents
                )

            cand_subsequent_common_table_names = reduce(
                operator.__or__, cand_subsequent_table_names.keys(), set()
            )

            for common_table_name in cand_subsequent_common_table_names:
                can_remove = True
                for common_table_name_set in cand_subsequent_table_names.keys():
                    if len(common_table_name_set - set([common_table_name])) == 0:
                        can_remove = False
                        break
                if can_remove:
                    self.log.log(
                        9,
                        f"remove {common_table_name} from {list(cand_subsequent_table_names.keys())}",
                    )
                    _cand_subsequent_table_names = {}
                    for common_table_name_set in cand_subsequent_table_names.keys():
                        new_key = common_table_name_set - set([common_table_name])
                        _cand_subsequent_table_names[new_key] = (
                            _cand_subsequent_table_names.get(new_key, set())
                            | cand_subsequent_table_names[common_table_name_set]
                        )
                    cand_subsequent_table_names = _cand_subsequent_table_names
            subsequent_table_names = cand_subsequent_table_names

            # filter predicates for the current target table
            (
                matched_preds,
                rest_of_preds,
            ) = CardEst._extract_matched_predicates_by_tables(
                predicates, used_table_names
            )
            predicates = rest_of_preds

            # list fanout cols to estimate
            fanout_cols = self._find_fanout_cols(
                table,
                subsequent_table_names,
            )

            # list cols shared with subsequenst joined tables
            common_cols = [col for col in fanout_cols.values()]
            for common_table_name_set in subsequent_table_names.keys():
                for common_table_name in common_table_name_set:
                    common_cols.extend(
                        [
                            col
                            for col in table.cols
                            if (
                                col.name.startswith(f"{common_table_name}.")
                                and not col.is_fanout()
                            )
                            or col.name == f"__in__:{common_table_name}"
                        ]
                    )

            self.log.log(9, f"θ on {[p.c for p in matched_preds]}")
            self.log.log(9, f"common: {list(common_samples.keys())}")
            (probs, fanouts, common_samples) = self._estimate(
                table,
                matched_preds,
                sample_size,
                self.model[u],
                # need `repeat` to copy tensors
                self.init_samples[u].repeat((sample_size, 1)),
                self.init_inputs[u].repeat((sample_size, 1)),
                self.init_logits[u].repeat((sample_size, 1)),
                fanout_cols=fanout_cols,
                inherited_samples=common_samples,
                common_cols=common_cols,
            )

            # common_samples = {}  # DEBUG: disable common samples

            cards *= probs
            self.log.debug(
                f"* P({u}) (≈{probs.min().item():.3f},{probs.median().item():.3f},{probs.max().item():.3f})"
            )

            for ts, fo in fanouts.items():
                # join fanout scaling
                cards *= fo
                self.log.debug(
                    f"* F({ts[0]}->{ts[1]}) (≈{fo.min().item():.3f},{fo.median().item():.3f},{fo.max().item():.3f})"
                )

        return cards

    def _find_fanout_cols(
        self,
        table: scardina.common.CsvTable,
        #                        { common_set: centers }
        subsequent_table_names: Dict[Set[str], Set[str]],
    ):
        # NOTE: In PJ, since all not used tables are connected to center table w/ 1-to-n,
        #       scaling by not used tables is always 1.

        fanout_cols = {}  # Dict[(table_from, table_to), col]
        for col in table.cols:
            if col.is_fanout():
                # for scaling by joins with subsequent tables
                for common_set, centers in subsequent_table_names.items():
                    for common in common_set:
                        for center in centers:
                            if (
                                re.match(
                                    rf"^{common}\.__adj_fanout__:[^=]+={center}\.",
                                    col.name,
                                )
                                is not None
                            ):
                                assert (
                                    len(col.scols) == 1
                                )  # contfanout or not factorized
                                fanout_cols[(common, center)] = col
        return fanout_cols

    def query(self, query):
        with torch.inference_mode():
            self.on_start()
            cards = self._estimate_over_models(
                self.sample_size,
                query,
            )
            card = cards.mean().item()
            elapsed_time_ms = self.on_end()

            return (
                np.ceil(card).astype(dtype=np.int, copy=False).item(),
                elapsed_time_ms,
            )
