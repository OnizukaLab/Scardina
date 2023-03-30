"""Dataset registrations."""
import os
import glob
from typing import Dict

import ray
import numpy as np
import pandas as pd
import pyarrow as pa
import networkx as nx

import scardina.common
import scardina.join
import scardina.schema


class Loader:
    def __init__(
        self,
        fact_threshold: int,
        join_sample_size_min: int,
        join_sample_size_max: int,
        device: str,
        dataset_root_path: str = "datasets",
    ):
        self.fact_threshold = fact_threshold
        self.n_join_samples_min = join_sample_size_min
        self.n_join_samples_max = join_sample_size_max
        self.device = device
        self.dataset_root_path = dataset_root_path

    def load(
        self, dataset_name: str, relation_type: str, cache_dir: str
    ) -> scardina.common.DB:
        if dataset_name == "dmv":
            return self.load_dmv(cache_dir)
        elif dataset_name == "dmv-tiny":
            return self.load_dmv(cache_dir, file_name="dmv-tiny.csv")
        elif dataset_name == "dmv-1":
            return self.load_dmv_1()
        elif dataset_name == "dmv-2":
            return self.load_dmv_2()
        elif dataset_name == "dmv-5":
            return self.load_dmv_5()
        elif dataset_name == "flight-delays":
            return self.load_flight_delays()
        elif dataset_name == "flight-delays-tiny":
            return self.load_flight_delays("flight-delays-tiny.csv")
        elif dataset_name == "imdb":
            if relation_type == "cin":
                return self.load_imdb_pj(cache_dir)
            elif relation_type == "ur":
                return self.load_imdb(cache_dir)
            else:
                raise ValueError(f"Unexpected relation type: {relation_type}")
        elif dataset_name == "imdb-job-light":
            if relation_type == "cin":
                return self.load_imdb_job_light_pj(cache_dir)
            elif relation_type == "ur":
                return self.load_imdb_job_light(cache_dir)
            else:
                raise ValueError(f"Unexpected relation type: {relation_type}")
        elif dataset_name == "imdb-tiny":
            return self.load_imdb("imdb-1m.csv")
        else:
            raise ValueError(f"Unexpected dataset name: {dataset_name}")

    def _prepare_data(
        self,
        schema: scardina.schema.Schema,
        dataset_dir: str,
    ):
        table_names = schema.get_table_names()

        found_all_cache = True
        for _, subschema_g in schema.subschemas.items():
            found_all_cache &= os.path.exists(subschema_g.graph["file_path"])
        if found_all_cache:
            return

        cache_dir = schema.g.graph["cache_dir"]

        # calc fanout values for each foreign key and save them as pickle
        @ray.remote
        def load_table_and_calc_fanout(
            table_name, fk_col_names, *, include_col_names=None, parse_dates=False
        ):
            print(f"S calculating fanout: {table_name}...")

            file_path = os.path.join(dataset_dir, f"{table_name}.csv")
            table = pd.read_csv(
                file_path, usecols=include_col_names, parse_dates=parse_dates, escapechar="\\"
            )
            table = table.convert_dtypes()

            # drop tz
            # Currently, expecting UTC only
            for col_name in table.columns:
                col = table[col_name]
                if col.dtype.name.startswith("datetime") and col.dt.tz is not None:
                    print(
                        f"Info: drop tz ({col.dtype.name}, {col.dt.tz.zone}) from {table_name}.{col_name}"
                    )
                    table[col_name] = col.dt.tz_localize(None)

            for fk_col_name in fk_col_names:
                file_path = os.path.join(
                    cache_dir, "fk_counts", f"{table_name}.{fk_col_name}.pickle"
                )
                if os.path.exists(file_path):
                    continue  # already cached

                fk_counts = table[fk_col_name].value_counts().rename("count")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                fk_counts.to_pickle(file_path)

            print(f"E calculating fanout: {table_name}...")
            return table

        futures = [
            load_table_and_calc_fanout.remote(
                table_name,
                sum(
                    [
                        [v[table_name] for v in schema.g.get_edge_data(*e).values()]
                        for e in schema.g.in_edges(table_name)
                    ],
                    [],
                ),
                include_col_names=schema.g.nodes[table_name]["col_names"],
                parse_dates=[
                    c
                    for c, t in schema.g.nodes[table_name]["type_casts"].items()
                    if t == np.datetime64
                ],
            )
            for table_name in table_names
        ]
        ray.wait(futures, num_returns=len(futures))
        data_object_ids = dict(zip(table_names, futures))

        # match fanout to a *single* opposite table and save only matched fanout col
        # (for parallel execution)
        @ray.remote
        def match_fanout(table1, table2, join):
            print(f"S matching fanout: {join}...")
            t1, t2, _ = join
            c1 = schema.g.edges[join][t1]
            c2 = schema.g.edges[join][t2]
            fk_counts1_path = os.path.join(cache_dir, "fk_counts" f"{t1}.{c1}.pickle")
            fk_counts2_path = os.path.join(cache_dir, "fk_counts", f"{t2}.{c2}.pickle")
            fo1_name = f"__adj_fanout__:{c2}={t1}.{c1}"
            fo2_name = f"__adj_fanout__:{c1}={t2}.{c2}"
            t1_fanout_file_path = os.path.join(
                cache_dir, "fanouts", f"{t1}.{fo2_name}.pickle"
            )
            t2_fanout_file_path = os.path.join(
                cache_dir, "fanouts", f"{t2}.{fo1_name}.pickle"
            )

            if not os.path.exists(t1_fanout_file_path):
                if os.path.exists(fk_counts2_path):
                    fo2 = pd.read_pickle(fk_counts2_path)
                    table1 = table1.merge(
                        fo2, left_on=c1, right_index=True, how="outer"
                    )
                    table1.rename(columns={"count": fo2_name}, inplace=True)
                    table1.fillna({fo2_name: 1}, inplace=True)
                    table1 = table1.astype({fo2_name: np.int64})
                else:  # fk table
                    table1[fo2_name] = 1
                os.makedirs(os.path.dirname(t1_fanout_file_path), exist_ok=True)
                table1[fo2_name].to_pickle(t1_fanout_file_path)

            if not os.path.exists(t2_fanout_file_path):
                if os.path.exists(fk_counts1_path):
                    fo1 = pd.read_pickle(fk_counts1_path)
                    table2 = table2.merge(
                        fo1, left_on=c2, right_index=True, how="outer"
                    )
                    table2.rename(columns={"count": fo1_name}, inplace=True)
                    table2.fillna({fo1_name: 1}, inplace=True)
                    table2 = table2.astype({fo1_name: np.int64})
                else:  # fk table
                    table2[fo1_name] = 1
                os.makedirs(os.path.dirname(t2_fanout_file_path), exist_ok=True)
                table2[fo1_name].to_pickle(t2_fanout_file_path)

            print(f"E matching fanout: {join}...")

        futures = [
            match_fanout.remote(
                data_object_ids[join[0]], data_object_ids[join[1]], join
            )
            for join in schema.g.edges
        ]
        ray.wait(futures, num_returns=len(futures))

        # add all adjacencies' fanout to each table and save whole table
        @ray.remote
        def merge_fanout_and_extract_distinct_vals(table, table_name):
            print(f"S merging fanout: {table_name}...")
            file_path = os.path.join(
                cache_dir, "base_tables_with_fanouts", f"{table_name}.pickle"
            )
            if os.path.exists(file_path):
                table = pd.read_pickle(file_path)
            else:
                fo_files_name = glob.glob(
                    os.path.join(
                        cache_dir, "fanouts", f"{table_name}.__adj_fanout__:*.pickle"
                    )
                )
                for fo_file_name in fo_files_name:
                    fo = pd.read_pickle(fo_file_name)
                    table = table.join(fo)

                # deterministic col order for compat
                table.sort_index(axis="columns", inplace=True)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                table.to_pickle(file_path)

            # extract distinct values
            for col_name in table.columns:
                file_path = os.path.join(
                    cache_dir, "distinct_vals", f"{table_name}.{col_name}.npy"
                )
                if os.path.exists(file_path):
                    continue

                # unique
                distinct_vals = table[col_name].unique()
                # non-null
                if type(distinct_vals) == np.ndarray:
                    distinct_vals = distinct_vals[~pd.isnull(distinct_vals)]
                else:
                    distinct_vals = distinct_vals.dropna()
                    # to ndarray
                    dtype = distinct_vals.dtype.name
                    if dtype == "string":
                        dtype = "object"
                    distinct_vals = distinct_vals.to_numpy(dtype=dtype)
                try:
                    distinct_vals.sort()
                except Exception as e:
                    print(e)
                    print(table)
                    print(table[col_name])
                    print(table[col_name].value_counts())
                    print(f"{col_name}, {type(distinct_vals)}: {distinct_vals[:10]}")

                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.save(file_path, distinct_vals)

            print(f"E merging fanout: {table_name}...")
            return table

        futures = [
            merge_fanout_and_extract_distinct_vals.remote(
                data_object_ids[table_name], table_name
            )
            for table_name in table_names
        ]
        ray.wait(futures, num_returns=len(futures))
        data_object_ids = dict(zip(table_names, futures))

        # set table data ids to subschema
        for _, subschema_g in schema.subschemas.items():
            for node in subschema_g.nodes:
                subschema_g.nodes[node]["data_object_id"] = data_object_ids[node]

        # join sampling and add metadata
        @ray.remote
        def join_sampling(
            joined_table_name, subschema_g, n_join_samples_min, n_join_samples_max
        ):
            for node in subschema_g.nodes:
                if node != subschema_g.graph["center_table_name"]:
                    root_t = node
                    break

            sample_size = max(
                n_join_samples_min, min(n_join_samples_max, subschema_g.graph["size"])
            )
            print(f"S join sampling: {joined_table_name} (size: {sample_size})...")

            if os.path.exists(subschema_g.graph["file_path"]):
                print(f"E join sampling: {subschema_g.graph['file_path']}... (cached)")
                return

            for node in subschema_g.nodes:
                subschema_g.nodes[node]["data"] = ray.get(
                    subschema_g.nodes[node]["data_object_id"]
                ).copy()

            # calc weight (bottom-up)
            for src, dsts in reversed(
                list(nx.bfs_successors(subschema_g.to_undirected(), root_t))
            ):
                src_table = subschema_g.nodes[src]["data"]
                assert "__weight__" not in src_table
                src_table["__weight__"] = 1
                for dst in dsts:
                    dst_table = subschema_g.nodes[dst]["data"]
                    if "__weight__" not in dst_table:  # leaf
                        dst_table["__weight__"] = 1

                    # get join keys
                    join_edges = []
                    if isinstance(subschema_g, nx.MultiDiGraph):
                        edges = subschema_g.get_edge_data(src, dst)
                        if edges is not None:
                            for _, join_edge in edges.items():
                                join_edges.append(join_edge)
                        else:
                            edges = subschema_g.get_edge_data(dst, src)
                            for _, join_edge in edges.items():
                                join_edges.append(join_edge)
                    else:
                        if (src, dst) in subschema_g.edges:
                            join_edges.append(subschema_g.edges[(src, dst)])
                        else:
                            join_edges.append(subschema_g.edges[(dst, src)])

                    for join_edge in join_edges:
                        src_key = join_edge[src]
                        dst_key = join_edge[dst]
                        dst_fanout_col = f"__fanout__:{src_key}={dst}.{dst_key}"

                        # sum of each id
                        dst_fanout = dst_table.groupby(dst_key)["__weight__"].sum()
                        # to fanout col name
                        dst_fanout.rename(dst_fanout_col, inplace=True)

                        # outer-join scale factor into src table
                        src_table = src_table.merge(
                            dst_fanout, left_on=src_key, right_index=True, how="outer"
                        )
                        src_table.reset_index(drop=True, inplace=True)
                        src_table.fillna(
                            {dst_fanout_col: 1, "__weight__": 1}, inplace=True
                        )  # no opposite rows as 1
                        src_table = src_table.astype({dst_fanout_col: np.int64})
                        src_table["__weight__"] *= src_table[dst_fanout_col]
                        dst_table = dst_table.convert_dtypes()
                        subschema_g.nodes[dst]["data"] = dst_table

                # fill NAs, which were inserted by subsequent outer-joins, with 1
                for col in src_table.columns:
                    if col.startswith("__fanout__:") or col.startswith(
                        "__adj_fanout__:"
                    ):
                        src_table.fillna({col: 1}, inplace=True)

                # set the result w/ weight and fanout values
                src_table = src_table.convert_dtypes()
                subschema_g.nodes[src]["data"] = src_table

            joined = scardina.join.sample(subschema_g, root_t, sample_size)

            # deterministic col order for compat
            joined.sort_index(axis="columns", inplace=True)
            os.makedirs(os.path.dirname(subschema_g.graph["file_path"]), exist_ok=True)
            joined.to_pickle(subschema_g.graph["file_path"])

            print(f"E join sampling: {subschema_g.graph['file_path']}...")
            return

        futures = [
            join_sampling.remote(
                joined_table_name,
                subschema_g,
                self.n_join_samples_min,
                self.n_join_samples_max,
            )
            for joined_table_name, subschema_g in schema.subschemas.items()
        ]
        ray.wait(futures, num_returns=len(futures))

        return

    def _load_data_into_db(self, schema: scardina.schema.Schema):
        tables: Dict[str, scardina.common.CsvTable] = {}
        for joined_table_name, subschema_g in schema.subschemas.items():
            distinct_vals_dict = {}
            distinct_vals_file_paths = sum(
                [
                    glob.glob(
                        os.path.join(
                            schema.g.graph["cache_dir"], "distinct_vals", f"{n}.*.npy"
                        )
                    )
                    for n in subschema_g.nodes()
                ],
                [],
            )

            # on pk-fk pairs whose domains are the same, fk cols had been dropped at a merge phase
            for distinct_vals_file_path in distinct_vals_file_paths:
                col_name = os.path.splitext(os.path.basename(distinct_vals_file_path))[
                    0
                ]
                distinct_vals_dict[col_name] = np.load(
                    distinct_vals_file_path, allow_pickle=True
                )

            tables[joined_table_name] = scardina.common.CsvTable(
                joined_table_name,
                subschema_g.graph["file_path"],
                None,  # all cols
                self.fact_threshold,
                self.device,
                type_casts=subschema_g.graph["type_casts"],
                distinct_vals_dict=distinct_vals_dict,
                n_rows=subschema_g.graph["size"],
            )

        return scardina.common.DB(
            name=schema.g.graph["name"],
            vtable=None,
            tables=tables,
            schema=schema,
        )

    def load_dmv(self, cache_dir_root: str, *, file_name="dmv.csv"):
        csv_file = os.path.join(self.dataset_root_path, "dmv-ur", file_name)
        col_names = [
            "Record Type",
            "Registration Class",
            "State",
            "County",
            "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = scardina.common.CsvTable(
            "dmv-universal",
            csv_file,
            col_names,
            self.fact_threshold,
            self.device,
            type_casts=type_casts,
        )

        schema = scardina.schema.Schema("dmv", "dmv", cache_dir_root)
        schema.add_table("dmv", 1, col_names=col_names)
        return scardina.common.DB(name="dmv", vtable=table, tables={}, schema=schema)

    def load_dmv_1(self, file_name="dmv.csv"):
        csv_file = os.path.join(self.dataset_root_path, "dmv-ur", file_name)
        col_names = [
            # "Record Type",
            "Registration Class",
            "State",
            "County",
            "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = scardina.common.CsvTable(
            "dmv",
            csv_file,
            col_names,
            self.fact_threshold,
            self.device,
            type_casts,
        )
        return scardina.common.DB(name="dmv", vtable=table, tables={})

    def load_dmv_2(self, file_name="dmv.csv"):
        csv_file = os.path.join(self.dataset_root_path, "dmv-ur", file_name)
        col_names = [
            # "Record Type",
            "Registration Class",
            "State",
            # "County",
            "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = scardina.common.CsvTable(
            "dmv",
            csv_file,
            col_names,
            self.fact_threshold,
            self.device,
            type_casts,
        )
        return scardina.common.DB(name="dmv", vtable=table, tables={})

    def load_dmv_5(self, file_name="dmv.csv"):
        csv_file = os.path.join(self.dataset_root_path, "dmv-ur", file_name)
        col_names = [
            # "Record Type",
            # "Registration Class",
            # "State",
            # "County",
            # "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = scardina.common.CsvTable(
            "dmv",
            csv_file,
            col_names,
            self.fact_threshold,
            self.device,
            type_casts=type_casts,
        )
        return scardina.common.DB(name="dmv", vtable=table, tables={})

    def load_flight_delays(self, file_name: str = "flight-delays.csv"):
        csv_file = os.path.join(self.dataset_root_path, "dmv-ur", file_name)
        col_names = [
            "YEAR_DATE",
            "UNIQUE_CARRIER",
            "ORIGIN",
            "ORIGIN_STATE_ABR",
            "DEST",
            "DEST_STATE_ABR",
            "DEP_DELAY",
            "TAXI_OUT",
            "TAXI_IN",
            "ARR_DELAY",
            "AIR_TIME",
            "DISTANCE",
        ]
        type_casts = {"YEAR_DATE": pa.float32()}
        table = scardina.common.CsvTable(
            "flight_delays",
            csv_file,
            col_names,
            self.fact_threshold,
            self.device,
            type_casts,
        )
        return scardina.common.DB(
            name="FlightDelays", vtable=table, tables={table.name: table}
        )

    def load_imdb(self, cache_dir_root: str):
        schema = self._build_imdb_job_light_schema(cache_dir_root)
        schema.build_ur_subschema_graph(11244784701195)
        self._prepare_data(schema, os.path.join(self.dataset_root_path, "imdb"))
        return self._load_data_into_db(schema)

    def load_imdb_job_light(self, cache_dir_root: str):
        schema = self._build_imdb_job_light_schema(cache_dir_root)
        schema.build_ur_subschema_graph(2128877229383)
        self._prepare_data(schema, os.path.join(self.dataset_root_path, "imdb"))
        return self._load_data_into_db(schema)

    def _build_imdb_schema(self, cache_dir_root: str) -> scardina.schema.Schema:
        schema = scardina.schema.Schema("imdb", "imdb", cache_dir_root)

        schema.add_table(
            "kind_type",
            7,
            col_names=[
                "id",
                "kind",
            ],
        )
        schema.add_table(
            "title",
            2528313,
            col_names=[
                "id",
                "title",
                "imdb_index",
                "kind_id",
                "production_year",
                # "imdb_id",
                "phonetic_code",
                # "episode_of_id",
                "season_nr",
                "episode_nr",
                "series_years",
                # "md5sum",
            ],
        )
        schema.add_table(
            "movie_companies",
            2609129,
            col_names=[
                # "id",
                "movie_id",
                "company_id",
                "company_type_id",
                "note",
            ],
        )
        schema.add_table(
            "company_name",
            234997,
            col_names=[
                "id",
                "name",
                "country_code",
                # "imdb_id",
                # "name_pcode_nf",
                # "name_pcode_sf",
                # "md5sum",
            ],
        )
        schema.add_table(
            "company_type",
            4,
            col_names=[
                "id",
                "kind",
            ],
        )
        schema.add_table(
            "aka_title",
            361472,
            col_names=[
                # "id",
                "movie_id",
                # "title",
                # "imdb_index",
                # "kind_id",
                # "production_year",
                # "phonetic_code",
                # "episode_of_id",
                # "season_nr",
                # "episode_nr",
                # "note",
                # "md5sum",
            ],
        )
        schema.add_table(
            "cast_info",
            36244344,
            col_names=[
                # "id",
                "person_id",
                "movie_id",
                "person_role_id",
                "note",
                "nr_order",
                "role_id",
            ],
        )
        schema.add_table(
            "movie_info",
            14835720,
            col_names=[
                # "id", #
                "movie_id",
                "info_type_id",
                "info",
                "note",
            ],
            type_casts={"note": pa.string()},
        )
        schema.add_table(
            "movie_info_idx",
            1380035,
            col_names=[
                # "id",
                "movie_id",
                "info_type_id",
                "info",
                # "note",
            ],
            type_casts={"info": pa.string()},
        )
        schema.add_table(
            "info_type",
            113,
            col_names=[
                "id",
                "info",
            ],
        )
        schema.add_table(
            "complete_cast",
            135086,
            col_names=[
                # "id",
                "movie_id",
                "subject_id",
                "status_id",
            ],
        )
        schema.add_table(
            "comp_cast_type",
            4,
            col_names=[
                "id",
                "kind",
            ],
        )
        schema.add_table(
            "movie_keyword",
            4523930,
            col_names=[
                # "id",
                "movie_id",
                "keyword_id",
            ],
        )
        schema.add_table(
            "keyword",
            134170,
            col_names=[
                "id",
                "keyword",
                "phonetic_code",
            ],
        )
        schema.add_table(
            "movie_link",
            29997,
            col_names=[
                # "id",
                "movie_id",
                "linked_movie_id",
                "link_type_id",
            ],
        )
        schema.add_table(
            "link_type",
            18,
            col_names=[
                "id",
                "link",
            ],
        )

        schema.add_relationship("kind_type", "id", "title", "kind_id")
        schema.add_relationship("title", "id", "aka_title", "movie_id")
        schema.add_relationship("title", "id", "movie_companies", "movie_id")
        schema.add_relationship("title", "id", "cast_info", "movie_id")
        schema.add_relationship("title", "id", "movie_info", "movie_id")
        schema.add_relationship("title", "id", "movie_info_idx", "movie_id")
        schema.add_relationship("title", "id", "complete_cast", "movie_id")
        schema.add_relationship("title", "id", "movie_keyword", "movie_id")
        schema.add_relationship("title", "id", "movie_link", "movie_id")
        schema.add_relationship("company_name", "id", "movie_companies", "company_id")
        schema.add_relationship(
            "company_type", "id", "movie_companies", "company_type_id"
        )
        schema.add_relationship("info_type", "id", "movie_info_idx", "info_type_id")
        schema.add_relationship("comp_cast_type", "id", "complete_cast", "subject_id")
        schema.add_relationship("keyword", "id", "movie_keyword", "keyword_id")
        schema.add_relationship("link_type", "id", "movie_link", "link_type_id")

        return schema

    def load_imdb_pj(self, cache_dir_root: str):
        schema = self._build_imdb_schema(cache_dir_root)
        subschema_sizes = {
            # TMP: 取り直したやつ
            "title:kind_id=kind_type.id": 2528314,
            "aka_title:movie_id=title.id": 2684154,
            "cast_info:movie_id=title.id": 36441056,
            "complete_cast:movie_id=title.id:subject_id=comp_cast_type.id": 2569887,
            "movie_companies:company_id=company_name.id:company_type_id=company_type.id:movie_id=title.id": 4050208,
            "movie_info:movie_id=title.id": 14895208,
            "movie_info_idx:info_type_id=info_type.id:movie_id=title.id": 3448531,
            "movie_keyword:keyword_id=keyword.id:movie_id=title.id": 6575449,
            "movie_link:link_type_id=link_type.id:movie_id=title.id": 2551901,
            # "title:kind_id=kind_type.id": 2528313,
            # "aka_title:movie_id=title.id": 361472,
            # "cast_info:movie_id=title.id": 36244344,
            # "complete_cast:movie_id=title.id:subject_id=comp_cast_type.id": 135086,
            # "movie_companies:company_id=company_name.id:company_type_id=company_type.id:movie_id=title.id": 2609129,
            # "movie_info:movie_id=title.id": 14835720,
            # "movie_info_idx:info_type_id=info_type.id:movie_id=title.id": 1380035,
            # "movie_keyword:keyword_id=keyword.id:movie_id=title.id": 4523930,
            # "movie_link:link_type_id=link_type.id:movie_id=title.id": 29997,
        }

        schema.build_subschema_graphs(subschema_sizes)
        self._prepare_data(schema, os.path.join(self.dataset_root_path, "imdb"))
        return self._load_data_into_db(schema)

    def _build_imdb_job_light_schema(
        self, cache_dir_root: str
    ) -> scardina.schema.Schema:
        schema = scardina.schema.Schema("imdb-job-light", "imdb", cache_dir_root)

        schema.add_table(
            "title",
            2528313,
            col_names=[
                "id",
                # "title",
                # "imdb_index",
                "kind_id",
                "production_year",
                # "phonetic_code",
                # "season_nr",
                # "episode_nr",
                # "series_years",
            ],
        )
        schema.add_table(
            "movie_companies",
            2609129,
            col_names=[
                "movie_id",
                "company_id",
                "company_type_id",
                # "note",
            ],
        )
        schema.add_table(
            "cast_info",
            36244344,
            col_names=[
                # "person_id",
                "movie_id",
                # "person_role_id",
                # "note",
                # "nr_order",
                "role_id",
            ],
        )
        schema.add_table(
            "movie_info",
            14835720,
            col_names=[
                "movie_id",
                "info_type_id",
                # "info",
                # "note",
            ],
            type_casts={"note": pa.string()},
        )
        schema.add_table(
            "movie_info_idx",
            1380035,
            col_names=[
                "movie_id",
                "info_type_id",
                # "info",
            ],
            type_casts={"info": pa.string()},
        )
        schema.add_table(
            "movie_keyword",
            4523930,
            col_names=[
                "movie_id",
                "keyword_id",
            ],
        )

        schema.add_relationship("title", "id", "movie_companies", "movie_id")
        schema.add_relationship("title", "id", "cast_info", "movie_id")
        schema.add_relationship("title", "id", "movie_info", "movie_id")
        schema.add_relationship("title", "id", "movie_info_idx", "movie_id")
        schema.add_relationship("title", "id", "movie_keyword", "movie_id")

        return schema

    def load_imdb_job_light_pj(self, cache_dir_root: str):
        schema = self._build_imdb_job_light_schema(cache_dir_root)
        subschema_sizes = {
            "movie_companies:movie_id=title.id": 4050206,
            "cast_info:movie_id=title.id": 36441056,
            "movie_info:movie_id=title.id": 14895208,
            "movie_info_idx:movie_id=title.id": 3448423,
            "movie_keyword:movie_id=title.id": 6575449,
        }

        schema.build_subschema_graphs(subschema_sizes)
        self._prepare_data(schema, os.path.join(self.dataset_root_path, "imdb"))
        return self._load_data_into_db(schema)
