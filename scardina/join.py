import time
import pickle
import os

import psycopg2
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.csv
import ray
import networkx as nx


def fetch(tables, data, w_col, id_col):
    for i, t in enumerate(tables):
        if isinstance(t, list):
            fetch(t, data[i], w_col[i], id_col[i])
        else:
            if os.path.exists(f".cache/{t}.csv"):
                continue
            data[i] = pd.read_sql(f"select * from {t}", conn)
            print(f"fetched {t}")
            if w_col[i] is not None:
                sum_w = sum(data[i][w_col[i]])
                data[i][w_col[i]] = data[i][w_col[i]] / float(sum_w)
                data[i].to_csv(f".cache/{t}.csv", index=False)
                print(f"saved {t}")
            else:
                data[i].to_csv(f".cache/{t}.csv", index=False)
                print(f"saved {t}")


def read(tables, data):
    for i, t in enumerate(tables):
        if isinstance(t, list):
            read(t, data[i])
        else:
            if data[i] is not None:
                continue
            data[i] = pa.csv.read_csv(f".cache/{t}.csv").to_pandas()
            print(f"read {t}")


# @ray.remote(num_returns=2)
def calc_weight(each_fk, fks, w_col_i):
    ws_each_fk = {}
    idx_each_fk = {}
    print(f"fks: {fks}")
    for fk in fks:
        g = each_fk.get_group(fk)
        idx_each_fk[fk] = g.index.values
        if w_col_i is not None:
            ws_each_fk[fk] = g[w_col_i].values.astype(np.float)
            ws_each_fk[fk] /= ws_each_fk[fk].sum()  # to [0, 1]
        else:
            ws_each_fk[fk] = None
    return idx_each_fk, ws_each_fk


# @ray.remote
def weighted_sampling(idx_each_fk, ws_each_fk, fks):
    to_merge = []
    for fk in fks:
        cands = idx_each_fk.get(fk, None)
        if cands is not None:
            to_merge.append(np.random.choice(cands, p=ws_each_fk[fk]))
        else:
            # no row to join, so use a NaN row
            to_merge.append(-1)
    return to_merge


def sample(
    schema_g: nx.MultiDiGraph,
    root_table: str,
    sample_size: int,
) -> pd.DataFrame:
    cache_dir = os.path.join(schema_g.graph["cache_dir"], "idx_ws")
    agg = None
    for src, dsts in nx.bfs_successors(schema_g.to_undirected(), root_table):
        if agg is None:
            src_table = schema_g.nodes[src]["data"]
            agg = src_table.sample(
                sample_size, weights=src_table["__weight__"], replace=True
            )
            agg = agg.reset_index(drop=True).add_prefix(f"{src}.")
            agg[f"__in__:{src}"] = True
        # else: already merged as a dst table

        for dst in dsts:
            print(f"start merging {dst}...")
            t0 = time.time()

            many_to_one = (dst, src) in schema_g.edges
            dst_one_or_many = "O" if many_to_one else "M"
            edge_attrs_list = []
            if isinstance(schema_g, nx.MultiDiGraph):
                if many_to_one:
                    edges = schema_g.get_edge_data(dst, src)
                else:
                    edges = schema_g.get_edge_data(src, dst)
                for _, edge_attrs in edges.items():
                    edge_attrs_list.append(edge_attrs)
            else:
                edge_attrs_list.append(
                    schema_g.edges[(dst, src)]
                    if many_to_one
                    else schema_g.edges[(src, dst)]
                )
            for edge_attrs in edge_attrs_list:
                src_k_c = edge_attrs[src]
                dst_k_c = edge_attrs[dst]
                is_leaf = schema_g.degree[dst] == 1
                w_c = None if is_leaf else "__weight__"
                dst_table = schema_g.nodes[dst]["data"]

                to_merge = np.empty((sample_size,), dtype=np.int)
                idx_each_fk = {}
                ws_each_fk = {}
                idx_file_path = os.path.join(
                    cache_dir, f"{dst}.{dst_k_c}_{dst_one_or_many}_idx.pickle"
                )
                ws_file_path = os.path.join(
                    cache_dir, f"{dst}.{dst_k_c}_{dst_one_or_many}_ws.pickle"
                )
                if os.path.exists(idx_file_path) and os.path.exists(ws_file_path):
                    print("found cache idx and weights for each fkey")
                    with open(idx_file_path, "rb") as f:
                        idx_each_fk = pickle.load(f)
                    with open(ws_file_path, "rb") as f:
                        ws_each_fk = pickle.load(f)
                else:
                    dd = (
                        dst_table[[dst_k_c, w_c]]
                        if not is_leaf
                        else dst_table[[dst_k_c]]
                    )  # select only fkey and weight cols for calc efficiency
                    each_fk = dd.groupby(dst_k_c)  # not materialized?
                    print(f"{len(each_fk)} groups")  # materialized here?

                    for c, fk in enumerate(each_fk.groups):
                        if c % 50000 == 0:
                            t1 = time.time()
                            print(f"{c}: {t1-t0}")
                            t0 = t1

                        g = each_fk.get_group(fk)
                        idx_each_fk[fk] = g.index.values
                        if not is_leaf:
                            ws_each_fk[fk] = g[w_c].values.astype(np.float)
                            ws_each_fk[fk] /= ws_each_fk[fk].sum()  # to [0, 1]
                        else:
                            ws_each_fk[fk] = None

                    os.makedirs(cache_dir, exist_ok=True)
                    with open(idx_file_path, "wb") as f:
                        pickle.dump(idx_each_fk, f)
                    with open(ws_file_path, "wb") as f:
                        pickle.dump(ws_each_fk, f)
            print(f"preprocessed for {dst} in {time.time()-t0}sec.")

            t0 = time.time()
            ids = agg[f"{src}.{src_k_c}"]

            if not many_to_one:
                unique_ids = np.unique(ids)
                if len(unique_ids) > 10:
                    for i, _id in enumerate(ids):
                        if i % 500000 == 0:
                            t1 = time.time()
                            print(f"{i}: {t1-t0}")
                            t0 = t1
                        cands = idx_each_fk.get(_id, None)
                        if cands is not None:
                            to_merge[i] = np.random.choice(cands, p=ws_each_fk[_id])
                        else:
                            # no row to join, so use a NaN row
                            to_merge[i] = -1
                else:
                    # 各IDに関して十分な数のsamplingを先にまとめて行う
                    memo = {}
                    for i, _id in enumerate(unique_ids):
                        cands = idx_each_fk.get(_id, None)
                        if cands is not None:
                            memo[_id] = np.random.choice(
                                cands,
                                size=len(ids),
                                replace=True,
                                p=ws_each_fk[_id],
                            )
                        else:
                            memo[_id] = None
                        t1 = time.time()
                        print(f"memo id: {_id}, {t1-t0}")
                        t0 = t1
                    print("use memo...")
                    for i, _id in enumerate(ids):
                        if i % 500000 == 0:
                            t1 = time.time()
                            print(f"{i}: {t1-t0}")
                            t0 = t1
                        m = memo[_id]
                        if m is not None:
                            to_merge[i] = m[i]
                        else:
                            to_merge[i] = -1
            else:
                for i, _id in enumerate(ids):
                    if i % 500000 == 0:
                        t1 = time.time()
                        print(f"{i}: {t1-t0}")
                        t0 = t1
                    cand = idx_each_fk.get(_id, None)
                    if cand is not None:
                        assert len(cand) == 1
                        to_merge[i] = idx_each_fk[_id][0]
                    else:
                        # no row to join, so use a NaN row
                        to_merge[i] = -1

            dst_table = dst_table.add_prefix(f"{dst}.")
            dst_table[f"__in__:{dst}"] = True
            dummy_row = dst_table.iloc[0].copy()
            for k, v in dummy_row.iteritems():
                if "__fanout__:" in k or "__adj_fanout__:" in k:
                    dummy_row[k] = 1
                else:
                    dummy_row[k] = pd.NA
            dummy_row[f"__in__:{dst}"] = False
            dst_table = dst_table.append(dummy_row, ignore_index=True)

            print("picking rows by indices... ", end="")
            to_merge_df = dst_table.iloc[to_merge]
            dst_table.drop(dst_table.index[-1])  # drop a dummy row
            t1 = time.time()
            print(f"done {t1-t0}")
            t0 = t1
            print("concatenating... ", end="")
            agg = pd.concat(
                [agg, to_merge_df.reset_index(drop=True)],
                axis=1,
            ).convert_dtypes()
            t1 = time.time()
            print(f"done {t1-t0}")
    return agg


if __name__ == "__main__":
    ...
