from dataclasses import dataclass
import os
import re
import math
import time
import random
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils import data
import pytorch_lightning as pl

import scardina.schema


class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('Attr1').Fill(data, infer_dist=True)

    The passed-in 'data' is copied by reference.
    """

    def __init__(
        self,
        name: str,
        table_name: str,
        fact_threshold: int,
        device: int,
        *,
        distinct_vals=None,
        fact_offset: int = 0,
        fact_window: int = None,
    ):
        self.name = name
        self.table_name = (
            table_name if "." not in name else re.sub(r"\..+$", "", self.name)
        )
        self.orig_name = re.sub(r"^.+?:", "", self.name)

        self.col_idx: int = None  # idx over all cols

        self.device = device

        # Data related fields.
        self.data = None
        self.size = None
        self.distinct_vals = distinct_vals  # Never contains null
        self.to_val = None  # will be used for fanout scaling

        # factorization
        self.fact_threshold: int = fact_threshold
        self.fact_offset: int = fact_offset
        self.fact_window: int = fact_window
        self.is_factorized = None
        self.is_scol: bool = False
        self.orig_col: Column = None
        self.scols: List[Column] = None
        self.fact_idx: int = None  # idx in col, i.e. always 0, 1,...
        self.scol_idx: int = None  # idx over all cols' scols
        self.scol_idxes: List[int] = None  # holding scols' idx over all cols' scols
        self.m = None  # scol bridge between prev and curr
        self.factorized_orig_distinct_vals = None
        # scols value given by largest col value
        # Suppose scols[:-1] are already sampled,
        # if samples[:-1] > fact_upper[:-1],
        # samples[-1] must be less than fact_upper[-1].
        self.fact_upper: np.ndarray = None

    # special column markers
    def is_fanout(self):
        return "_fanout__" in self.name or self.name.endswith("__weight__")

    def is_tbl_existence(self):
        return "__in__:" in self.name

    def tbl_existence(self):
        assert self.is_tbl_existence()
        return self.name[7:]  # e.g., __in__:title -> title

    def set_dist(self, vals):
        """This is all the values this column will ever see."""
        if self.distinct_vals is not None:
            self.dist_size = len(self.distinct_vals) + 2  # +2 for MASK and NULL
            self.distinct_val_set = set(self.distinct_vals)
            print(f"skip set_dist on {self.name} ", end="")
            return

        # unique
        self.distinct_vals = vals.unique()
        # drop nulls once
        if type(self.distinct_vals) == np.ndarray:
            self.distinct_vals = self.distinct_vals[~pd.isnull(self.distinct_vals)]
        else:
            self.distinct_vals = self.distinct_vals.dropna()
            # to ndarray
            dtype = self.distinct_vals.dtype.name
            if dtype == "string":
                dtype = "object"
            self.distinct_vals = self.distinct_vals.to_numpy(dtype=dtype)
        self.distinct_vals.sort()
        self.distinct_val_set = set(self.distinct_vals)
        self.dist_size = len(self.distinct_vals) + 2  # +2 for MASK and NULL

        assert type(self.distinct_vals) == np.ndarray
        assert pd.isnull(self.distinct_vals).sum() == 0

    def set_to_val(self):
        assert self.distinct_vals is not None
        assert self.is_fanout()
        # require list of indices which excludes MASK
        self.to_val = torch.as_tensor(
            # [0]: reserved for MASK but never appears for fanout
            # [1]: reserved for NULL but never appears for fanout
            # [2]: first not nan val
            # [3]: second not nan val
            # [n+1]: n-th not nan val
            np.insert(self.distinct_vals.astype(np.float, copy=False), 0, [-1, -1]),
            device=self.device,
        )

    def set_data(self, data):
        assert self.data is None
        self.data = data
        self.size = len(self.data)

    # factorization flow
    # 1. get index of target value in distinct_vals
    # 2. add 2 to index (padding mask and null)
    # 3. project index for each sub col
    # 3.1. slice index as bits
    # 3.2. add 1 for 0 as mask
    def factorize(self):
        self.is_factorized = self.dist_size > 2**self.fact_threshold
        if not self.is_factorized:
            return [self]

        fact_upper = []
        scols = []
        n_scols = math.ceil(self.dist_size.bit_length() / self.fact_threshold)
        scol_bits = [math.floor(self.dist_size.bit_length() / n_scols)] * n_scols
        remainder_bits = self.dist_size.bit_length() - sum(scol_bits)
        for i in range(remainder_bits):
            scol_bits[i] += 1

        for i in range(n_scols):
            scol = Column(
                f"{self.name}:{i}",
                self.table_name,
                self.fact_threshold,
                self.device,
                fact_offset=sum(scol_bits[:i]),
                fact_window=scol_bits[i],
            )
            scol.is_scol = True
            scol.fact_idx = i
            scol.orig_col = self
            scol.is_factorized = True

            # TODO: remove factorized_orig_distinct_vals
            # dist_size contains MASK and NULL
            # since factored columns don't handle NULL, remove room for MASK by -1
            factorized_orig_distinct_vals = scol.project_values(
                np.arange(1, self.dist_size)
            )
            fact_upper.append(scol.project_values(self.dist_size - 1))
            # ≈ np.unique(factorized_orig_distinct_vals)
            largest_val = scol.project_values(np.arange(1, self.dist_size)).max()
            scol.distinct_vals = np.arange(1, largest_val + 1)
            scol.factorized_orig_distinct_vals = torch.as_tensor(
                factorized_orig_distinct_vals, device=self.device
            )
            assert (scol.factorized_orig_distinct_vals > 0).all()
            scol.dist_size = scol.distinct_vals.size + 1  # +1 for MASK

            assert type(scol.distinct_vals) == np.ndarray
            assert np.isnan(scol.distinct_vals).sum() == 0

            scols.append(scol)

        self.fact_upper = torch.as_tensor(fact_upper, device=self.device)
        return scols

    # project values of original column into values of subcolumn
    def project_values(self, values):
        assert self.is_scol, "project_values is only for factored columns"
        assert (
            isinstance(values, int) and values != 0
        ) or 0 not in values, "TODO: factored[values==0]=0 ?"
        return (
            (values >> self.fact_offset) & (2**self.fact_window - 1)
        ) + 1  # +1 means 0 for MASK in factored columns

    def discretize_values(self, values, drop_ood=False, drop_null=False):
        if isinstance(values, str) or not hasattr(values, "__len__"):
            values = np.array([values])
        elif isinstance(values, list):
            values = np.array(values)

        if drop_ood and drop_null:
            # since distinct_vals never contains null, null will be handled as out-of-domain in default
            # in_domain = np.isin(values, self.distinct_vals) # slow
            in_domain_and_not_null = np.vectorize(lambda v: v in self.distinct_val_set)(
                values
            )  # temporal alternative
            values = values[in_domain_and_not_null]
        elif drop_ood and not drop_null:
            # in_domain = np.isin(values, self.distinct_vals)
            in_domain_and_not_null = np.vectorize(lambda v: v in self.distinct_val_set)(
                values
            )  # temporal alternative
            is_null = pd.isnull(values)
            values = values[in_domain_and_not_null | is_null]
        elif not drop_ood and drop_null:
            is_null = pd.isnull(values)
            values = values[~is_null]

        # Since we cannot represent MASK in pre-discretized data format,
        # MASK never appears in values.
        #   not factored cols       or  cols which holds scols
        if (not self.is_factorized) or (self.is_factorized and self.scols is not None):
            # pd.Categorical() does not allow categories be passed in an array
            # containing np.nan.  It makes it a special case to return code -1
            # for NaN values and values not in distinct_vals as well.
            cate_ids = pd.Categorical(values, categories=self.distinct_vals).codes
            assert (
                # not fanout
                "__fanout__:" not in self.name
                and "__adj_fanout__:" not in self.name
            ) or (
                # when fanout, must not contain null
                (cate_ids >= 0).all()
            )

            # Since nan/nat cate_id is supposed to be 1 but pandas returns -1,
            # just add 2 to everybody.
            # 0 represents <MASK>: never appears in the original dataset
            # 1 represents <NULL>: originally -1 from pd.Categorical()
            cate_ids = cate_ids + 2
            cate_ids = cate_ids.astype(np.int32, copy=False)
        else:
            cate_ids = pd.Categorical(values, categories=self.distinct_vals).codes
            cate_ids = cate_ids + 1  # +1 for MASK
            cate_ids = cate_ids.astype(np.int32, copy=False)
        # 0 (MASK) never appears in the original dataset
        assert (cate_ids > 0).all()
        return cate_ids

    def discretize(self, release_data=False):
        """Transforms data values into integers using a Column's vocab.

        Returns:
            col_data: discretized version; an np.ndarray of type np.int32.
        """
        assert (
            not self.is_scol
        ), f"{self.name} is a factorized col. apply discretize to only orig col."

        assert self.data is not None

        print(f"{self.name}... ", end="")

        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values and values not in distinct_vals as well.
        cate_ids_list = []
        cate_ids = self.discretize_values(self.data)

        if release_data:
            del self.data

        # discretize scols
        if len(self.scols) > 1:
            # for i, scol in enumerate(self.scols):
            for scol in self.scols:
                factorized_data = scol.project_values(cate_ids)
                scol_data = pd.Series(factorized_data)
                scate_ids = scol.discretize_values(scol_data)

                # 0 (MASK) never appears in the original dataset
                assert (scate_ids > 0).all()
                cate_ids_list.append(scate_ids)
        else:
            # not factorized
            cate_ids_list.append(cate_ids)

        return cate_ids_list

    def __repr__(self):
        return f"Column({self.name}, |domain|={self.dist_size})[{self.col_idx}, {self.scol_idx}]"


class Table(object):
    def __init__(self, name, cols: List[Column], scols: List[Column]):
        self.name: str = name
        self.n_rows: int = self._validate_cardinality(cols)
        self.cols: List[Column] = cols
        self.scols: List[Column] = scols
        self.n_cols: int = len(self.cols)
        self.n_scols: int = len(self.scols)

        self.name_to_idx = {c.name: i for i, c in enumerate(self.cols)}

    def __repr__(self):
        return f"{self.name}"

    def __getitem__(self, col_name):
        return self.cols[self.col_name_to_idx(col_name)]

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [c.size for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def col_name_to_idx(self, col_name):
        """Returns index of column with the specified name."""
        assert col_name in self.name_to_idx
        return self.name_to_idx[col_name]


class CsvTable(Table):
    def __init__(
        self,
        name,
        file_name_or_df,
        col_names,
        fact_threshold,
        device,
        *,
        type_casts={},
        distinct_vals_dict={},
        hold_data=True,
        n_rows=None,
        **kwargs,
    ):
        self.name = name
        self.hold_data = hold_data
        self.device = device

        if isinstance(file_name_or_df, str):
            self.data = self._load(file_name_or_df, col_names, type_casts, **kwargs)
        else:
            assert isinstance(file_name_or_df, pd.DataFrame)
            self.data = file_name_or_df

        self.cols = self._build_columns(
            col_names,
            type_casts,
            fact_threshold,
            distinct_vals_dict,
        )
        self.scols = self._factorize()

        if not self.hold_data:
            del self.data

        super(CsvTable, self).__init__(name, self.cols, self.scols)

        if n_rows is not None:
            self.n_rows = n_rows

    def _load(self, file_name, cols, type_casts, **kwargs):
        print(f"Loading {self.name}...", end=" ")
        s = time.time()
        if file_name.endswith(".pickle"):
            data = pd.read_pickle(file_name)
        else:
            if cols is not None:
                args = {
                    "include_columns": cols,
                    "column_types": {
                        c: tc for c, tc in type_casts.items() if tc != np.datetime64
                    },
                }
            else:
                args = {
                    "column_types": {
                        c: tc for c, tc in type_casts.items() if tc != np.datetime64
                    },
                }
            data = pa.csv.read_csv(
                file_name,
                convert_options=pa.csv.ConvertOptions(**args),
                **kwargs,
            ).to_pandas(split_blocks=True, self_destruct=True, date_as_object=False)
            data = data.convert_dtypes()
        # Drop weight columns
        data = data.filter(regex="^((?!__weight__).)*$", axis="columns")
        print(f"done, took {(time.time() - s):.1f}s")
        return data

    def _build_columns(
        self,
        col_names,
        type_casts,
        fact_threshold,
        distinct_vals_dict={},
    ):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print(f"Parsing {self.name}... ", end=" ")
        s = time.time()
        for col_name, typ in type_casts.items():
            if col_name not in self.data:
                print(f"Warn: type cast for {col_name} ({typ}) is ignored")
                continue
            if typ == np.datetime64:
                assert (
                    self.data[col_name].dt.tz is None
                ), f"{col_name} has tz ({self.data[col_name].dt.tz.zone}). Currently, this system doesn't care about tz. Drop tz first."
                # Both infer_datetime_format and cache are critical for perf.
                self.data[col_name] = pd.to_datetime(
                    self.data[col_name], infer_datetime_format=True, cache=True
                )
            elif typ == int:
                self.data[col_name] = self.data[col_name].astype(int)

        if col_names is None:
            col_names = self.data.columns
        cols = []
        # Note: col_names が distinct_vals_dict の要求順である必要がある
        # e.g., [..., "title.id", ..., "movie_companies.movie_id", ...]
        for col_idx, col_name in enumerate(col_names):
            # pk table側で得られたdistinct_valsの外挿
            distinct_vals = distinct_vals_dict.get(col_name, None)
            col = Column(
                col_name,
                self.name,
                fact_threshold,
                self.device,
                distinct_vals=distinct_vals,
            )
            col.col_idx = col_idx

            if col.is_fanout():
                self.data[col_name].fillna(1, inplace=True)

            if self.hold_data:
                col.set_data(self.data[col_name])
            else:
                col.size = len(self.data[col_name])

            if not col.is_tbl_existence():  # normal columns
                col.set_dist(self.data[col_name])
            else:  # existence marker
                col.set_dist(pd.Series([False, True]))

            if col.is_fanout():
                col.set_to_val()
            cols.append(col)
        print(f"done, took {(time.time() - s):.1f}s")

        return cols

    def _factorize(self, discrete_tables=None):
        scols = []
        for col_idx, col in enumerate(self.cols):
            scol_idx_s = len(scols)
            _scols = col.factorize()
            scols.extend(_scols)
            scol_idx_e = len(scols)
            col.scols = _scols
            col.scol_idxes = list(range(scol_idx_s, scol_idx_e))  # increment
            for scol, scol_idx in zip(col.scols, col.scol_idxes):
                scol.col_idx = col.col_idx
                scol.scol_idx = scol_idx
        return scols


class DB:
    def __init__(
        self,
        name: str,
        vtable: CsvTable,
        tables: Dict[str, CsvTable],
        *,
        schema: scardina.schema.Schema = None,
    ):
        self.name = name
        self.vtable = vtable  # joined virtual table
        self.tables = tables  # each table
        self.schema = schema

    def __repr__(self):
        return f"{self.name}({self.tables})"


class DBDataset(data.Dataset):
    def __init__(
        self,
        db_name: str,
        schema: scardina.schema.Schema,
        table: CsvTable,
        cont_fanout: bool,
        fact_threshold: int,
        *,
        cache_dir_root: str = ".",
    ):
        super(DBDataset, self).__init__()

        cache_dir_path = (
            os.path.join(
                schema.g.graph["cache_dir"],
                "discretized",
                f"{fact_threshold}",
            )
            if schema is not None
            else os.path.join(
                cache_dir_root, db_name, "discretized", f"{fact_threshold}"
            )
        )
        os.makedirs(cache_dir_path, exist_ok=True)
        print(f"Discretizing {table.name}... ", end="")
        cache_file_path = os.path.join(cache_dir_path, f"{table.name}.pt")
        if os.path.exists(cache_file_path):
            print(f"found cache {table.name}")
            self.tuples = torch.load(cache_file_path)
            del table.data
            return

        s = time.time()
        # [n_full_rows, n_cols]
        discretized = []
        for col in table.cols:
            if not col.is_fanout() or not cont_fanout:
                discretized.extend(col.discretize(release_data=True))
            else:
                cont_vals = col.data.values.astype(int)
                assert np.all(cont_vals >= 1)
                log2_cont_vals = np.log2(cont_vals)  # log-transformed
                # add epsilon to make a diff from mask
                log2_cont_vals[log2_cont_vals == 0] += np.finfo(
                    log2_cont_vals.dtype
                ).eps
                assert np.all(log2_cont_vals > 0)
                discretized.extend([log2_cont_vals])
                del col.data  # hack for reducing memory usage

        del table.data  # hack for reducing memory usage

        # (very slow, high memory usage)
        self.tuples = torch.as_tensor(np.stack(discretized, axis=1))

        #      factorized           >= original
        assert self.tuples.shape[1] >= len(table.cols)

        torch.save(self.tuples, cache_file_path)

        print(f"done, took {(time.time() - s):.1f}s")

    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        db_name: str,
        schema: scardina.schema.Schema,
        table: CsvTable,
        batch_size: int,
        fact_threshold: int,
        *,
        cache_dir_root: str = ".",
        cont_fanout: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size

        dataset = DBDataset(
            db_name,
            schema,
            table,
            cont_fanout,
            fact_threshold,
            cache_dir_root=cache_dir_root,
        )

        val_dataset_size = min(int(len(dataset) / 10), 10000)
        tra_dataset_size = len(dataset) - val_dataset_size
        print(f"train size: {tra_dataset_size}, val size: {val_dataset_size}")

        self.tra_dataset, self.val_dataset = torch.utils.data.random_split(
            torch.utils.data.dataset.Subset(
                dataset,
                random.sample(
                    list(range(len(dataset))), tra_dataset_size + val_dataset_size
                ),
            ),
            [tra_dataset_size, val_dataset_size],
        )

        self._train_dataloader = None
        self._val_dataloader = None

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self._train_dataloader is None:
            self._train_dataloader = torch.utils.data.DataLoader(
                self.tra_dataset,
                batch_size=self.batch_size,
                num_workers=os.cpu_count(),
                pin_memory=True,
            )
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self._val_dataloader is None:
            self._val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=os.cpu_count(),
                pin_memory=True,
            )
        return self._val_dataloader


@dataclass(frozen=True)
class Predicate:
    c: str  # full column name (e.g., <table_name>.<column_name>)
    o: str  # operator
    v: Union[str, List]  # value
    dom_o: str = None  # dominant operator for factored columns
    no_ood: bool = False  # guarantee all v are in domain

    def __post_init__(self):
        object.__setattr__(self, "c_tbl", re.sub(r"\..+$", "", self.c))
        object.__setattr__(self, "c_col", re.sub(r"^.+\.", "", self.c))
