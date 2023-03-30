import os
from typing import Dict, List, Optional
import itertools

import pandas as pd
import networkx as nx
import hypernetx as hx


class Schema:
    def __init__(
        self,
        name: str,
        master_name: str,
        cache_dir_root: str,
        *,
        force_lower_case: bool = False,
    ):
        self.g: nx.MultiDiGraph = nx.MultiDiGraph(
            name=name,
            master_name=master_name,
            cache_dir=os.path.join(cache_dir_root, name),
        )
        self.name = name
        self.master_name = master_name
        self.subschemas: Dict[str, nx.MultiDiGraph] = {}
        self.subschema_hg: hx.Hypergraph = None
        self.force_lower_case = force_lower_case

    def add_table(
        self,
        table_name: str,
        size: int,
        *,
        col_names: List[str] = [],
        type_casts: Dict[str, type] = {},
        data: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            table_name: table name
            size: table size
            col_names: column names to be loaded. if empty, load all columns.
            type_casts: type cast definition for each columns. if empty, use types
                suggested by pandas.
            data: pd.DataFrame represents a table, which is including all column data
        """
        if self.force_lower_case:
            table_name = table_name.lower()
            col_names = [c.lower() for c in col_names]
            type_casts = {c.lower(): t for c, t in type_casts.items()}

        self.g.add_node(
            table_name,
            **{
                "size": size,
                "type_casts": type_casts,
                "col_names": col_names,
                "data": data,
            },
        )

    def add_relationship(
        self, pk_table_name: str, pk_col_name: str, fk_table_name: str, fk_col_name: str
    ):
        if self.force_lower_case:
            pk_table_name = pk_table_name.lower()
            pk_col_name = pk_col_name.lower()
            fk_table_name = fk_table_name.lower()
            fk_col_name = fk_col_name.lower()

        self.g.add_edge(
            pk_table_name,
            fk_table_name,
            **{
                pk_table_name: pk_col_name,
                fk_table_name: fk_col_name,
                "__join__": f"{pk_table_name}.{pk_col_name}={fk_table_name}.{fk_col_name}",
            },
        )

    def get_table_names(self) -> List[str]:
        return list(self.g.nodes)

    def build_subschema_graphs(
        self, subschema_sizes: Dict[str, int]  # center_table_name: size
    ) -> Dict[str, nx.DiGraph]:
        subschema_forest = nx.Graph()  # to construct join hypergraph
        for center_tn in self.g.nodes:
            # extract subschema: incoming tables
            subschema_adjacent_table_names = list(
                set(u for u, _ in self.g.in_edges(center_tn))
            )
            subschema_table_names = subschema_adjacent_table_names + [center_tn]

            if len(subschema_adjacent_table_names) == 0:
                continue

            # build subschema (here, subschema_g might contain parallel edges)
            subschema_g: nx.MultiDiGraph = self.g.subgraph(
                subschema_table_names
            ).copy()  # copy for making independent instance from overall schema

            # support parallel edges
            parallel_adjacencies = []
            for adjacent_tn in subschema_adjacent_table_names:
                parallel_adjacencies.append(
                    [
                        # pk_table  , pk_col        , fk_table , fk_col
                        (adjacent_tn, v[adjacent_tn], center_tn, v[center_tn])
                        for v in subschema_g.get_edge_data(
                            adjacent_tn, center_tn
                        ).values()
                    ]
                )

            for join_info_list in itertools.product(*parallel_adjacencies):
                # remove not contained edges from subschema_g
                # use _subschema_g as a unique subschema graph
                # finally, it'll become a DiGraph instead of MultiDiGraph
                _subschema_g = subschema_g.copy()
                for u, v, k in subschema_g.edges:
                    edge_data = _subschema_g.get_edge_data(u, v, k)
                    if (u, edge_data[u], v, edge_data[v]) not in join_info_list:
                        _subschema_g.remove_edge(u, v, k)
                _subschema_g = nx.DiGraph(_subschema_g)

                subschema_name_parts = []
                for join_info in join_info_list:
                    pk_tn = join_info[0]
                    pk_cn = join_info[1]
                    fk_tn = join_info[2]
                    fk_cn = join_info[3]

                    # type_casts for the center table
                    type_casts = {}
                    for col_name, ty in _subschema_g.nodes[fk_tn]["type_casts"].items():
                        type_casts[f"{fk_tn}.{col_name}"] = ty
                    type_casts[f"__in__:{fk_tn}"] = bool

                    # for edge_idx, adjacent_tn in enumerate(adjacencies):
                    # type_casts for adjacencies
                    for col_name, ty in _subschema_g.nodes[pk_tn]["type_casts"].items():
                        type_casts[f"{pk_tn}.{col_name}"] = ty
                    type_casts[f"__in__:{pk_tn}"] = bool

                    subschema_name_parts.append(f":{fk_cn}={pk_tn}.{pk_cn}")

                # set attributes for the subschema
                subschema_name = center_tn + "".join(sorted(subschema_name_parts))
                assert (
                    subschema_name in subschema_sizes
                ), f"size of subschema {subschema_name} is not registered"
                subschema_file_path = os.path.join(
                    _subschema_g.graph["cache_dir"],
                    "joined_tables",
                    f"{subschema_name}.pickle",
                )
                _subschema_g.graph["name"] = subschema_name
                _subschema_g.graph["center_table_name"] = center_tn
                _subschema_g.graph["file_path"] = subschema_file_path
                _subschema_g.graph["size"] = subschema_sizes[subschema_name]
                _subschema_g.graph["type_casts"] = type_casts

                self.subschemas[subschema_name] = _subschema_g

                # hypergraph-related
                # subjoin_nodes = set(_subschema_g.nodes)
                # add nodes
                # subschema_forest.add_nodes_from(subjoin_nodes, bipartite=0)
                subschema_forest.add_nodes_from(_subschema_g.nodes, bipartite=0)
                # add hyperedges
                subschema_forest.add_node(
                    subschema_name,
                    joined_table=_subschema_g,
                    bipartite=1,
                )
                # build hyperedges
                for node in _subschema_g.nodes:
                    subschema_forest.add_edge(node, subschema_name)

        self.subschema_hg = hx.Hypergraph.from_bipartite(subschema_forest)
        return self.subschemas

    def build_ur_subschema_graph(
        self, size: int  # univ rel size
    ) -> Dict[str, nx.DiGraph]:
        type_casts = {}
        for t in self.g.nodes:
            for col_name, ty in self.g.nodes[t]["type_casts"].items():
                type_casts[f"{t}.{col_name}"] = ty
            type_casts[f"__in__:{t}"] = bool

        self.g.graph["center_table_name"] = ""
        self.g.graph["file_path"] = os.path.join(
            self.g.graph["cache_dir"],
            "joined_tables",
            f"{self.g.graph['name']}.pickle",
        )
        self.g.graph["size"] = size
        self.g.graph["type_casts"] = type_casts

        self.subschemas[self.g.graph["name"]] = self.g
        return self.subschemas
