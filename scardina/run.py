import os
import re
import glob
import json
import random
import shutil
import pickle
import logging
import argparse
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import torch.backends.cudnn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
import ray
from ray import tune
from ray.tune.logger import LoggerCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import sqlparse as sp
from wandb import wandb
import rich.logging
import tqdm

import cardinality_estimation.algs
import query_representation.query
import evaluation.eval_fns

import scardina.estimators
import scardina.datasets
import scardina.common
import scardina.parse
import scardina.models



def calc_entropy(name, data, bases=None):
    import scipy.stats

    s = "Entropy of {}:".format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        s += " {:.4f} {}".format(e, unit)
    print(s)
    return ret


def calc_q_err(est_card, true_card):
    if true_card == 0 and est_card != 0:
        return est_card
    if true_card != 0 and est_card == 0:
        return true_card
    if true_card == 0 and est_card == 0:
        return 1.0
    return max(est_card / true_card, true_card / est_card)


def query(
    est: scardina.estimators.CardEst,
    query,
    true_card: int,
    log: logging.Logger,
    *,
    i=None,
):
    cols, ops, vals, tbls, joins, raw_sql = query

    conds = [f""""{c}" {o} '{str(v)}'""" for c, o, v in zip(cols, ops, vals)]
    pseudo_sql = f"select count(1) from {','.join(tbls)} where {' and '.join(conds)}"
    log.info(f"q{i}:\t{raw_sql}")
    log.debug(f"\t{pseudo_sql}")

    est_card, elapsed_time_ms = est.query(query[:-1])
    err = calc_q_err(est_card, true_card)
    est.add_err(err, est_card, true_card)
    log.info(
        f"actual: {true_card}, est: {est_card} (err={err:.3f}, {'↑' if est_card > true_card else '↓'})\n"
    )

    wandb.log(
        {
            "query_no": i,
            "q_err": err,
            "true_card": true_card,
            "est_card": est_card,
            "elapsed_time_ms": elapsed_time_ms,
            "sql": raw_sql,
            "pseudo sql": pseudo_sql,
        }
    )  # in term of wandb, `i` is used for `step` as well
    return est_card


default_model = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inc-train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--cache-data", action="store_true")

    parser.add_argument("--params-dir", type=str)
    parser.add_argument(
        "--model-type", "-t", type=str, choices=["trm", "mlp"], required=True
    )
    parser.add_argument(
        "--schema-strategy", "-s", type=str, choices=["cin", "ur"], default="cin"
    )
    parser.add_argument("--dataset", "-d", type=str, required=True)

    def nullable_int(val: str):
        if val.isdigit():
            return int(val)
        return None

    parser.add_argument("--n-trials", type=nullable_int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--n-blocks", type=int)  # for trm
    parser.add_argument("--n-heads", type=int)  # for trm
    parser.add_argument("--d-word", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--n-ff", type=int)  # for mlp
    parser.add_argument("--without-pos-emb", action="store_true")
    parser.add_argument("--fact-threshold", type=int, default=16)
    parser.add_argument("--smaller-emb", type=str, default="")
    parser.add_argument("--join-sample-size-min", type=int, default=100000)
    parser.add_argument("--join-sample-size-max", type=int, default=500000)

    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)

    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--benchmark", "-b", type=str)
    parser.add_argument("--eval-all-intermediate-models", "-A", action="store_true")
    parser.add_argument("--eval-sample-size", type=int, default=1000)
    parser.add_argument(
        "--eval-order",
        type=str,
        choices=["prop", "prop-ratio", "prop-inv", "domain-size", "nat", "inv"],
        default="prop-ratio",
    )
    parser.add_argument("--eval-disable-skip-high-card-cols", action="store_true")

    parser.add_argument("--cont-fanout", action="store_true")

    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--dump-intermediates", action="store_true")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--logging-level", type=str, default="info")
    args, unknown_args = parser.parse_known_args()

    unknown_args_without_empty = []
    for arg in unknown_args:
        if arg != "":
            unknown_args_without_empty.append(arg)
    assert (
        len(unknown_args_without_empty) == 0
    ), f"unknown argments: {unknown_args_without_empty}"

    logging_level: str = args.logging_level.upper()
    logging.basicConfig(handlers=[rich.logging.RichHandler()], format="%(message)s")
    logging.addLevelName(logging.DEBUG - 1, "TRACE")
    log = logging.getLogger("scardina")
    log.setLevel(logging_level)

    local_mode: bool = args.local
    do_train: bool = args.train
    # do_inc_train: bool = args.inc_train
    do_eval: bool = args.eval
    daemon_mode: bool = args.daemon
    # cache_data: bool = args.cache_data

    device: str = args.device
    assert device != "cuda" or torch.cuda.is_available()

    params_dir: str = args.params_dir
    model_type: str = args.model_type
    schema_strategy: str = args.schema_strategy
    dataset_name: str = args.dataset
    n_trials: Optional[int] = args.n_trials  # if not None, do hyperparam search
    warmups: Optional[int] = args.warmups

    assert not (
        params_dir is not None and n_trials is not None
    ), "params files can be used as config base only if not use hyperparam search"

    # to specify searchable params
    # will be used w/ `n_traial is None`
    d_word: Optional[int] = args.d_word
    n_blocks: Optional[int] = args.n_blocks
    d_ff: Optional[int] = args.d_ff
    n_ff: Optional[int] = args.n_ff
    n_heads: Optional[int] = args.n_heads
    batch_size: Optional[int] = args.batch_size
    lr: Optional[float] = args.lr

    # non-searchable params
    n_epochs: int = args.epochs
    fact_threshold: int = args.fact_threshold
    smaller_emb: str = args.smaller_emb
    cont_fanout: bool = args.cont_fanout
    join_sample_size_min: int = args.join_sample_size_min
    join_sample_size_max: int = args.join_sample_size_max
    assert join_sample_size_min <= join_sample_size_max

    model_path: str = args.model
    if model_path is None and do_eval:
        model_path = default_model[dataset_name][model_type][schema_strategy]
    eval_all_intermediate_models = args.eval_all_intermediate_models
    benchmark_name: str = args.benchmark
    eval_sample_size: int = args.eval_sample_size
    eval_order: str = args.eval_order
    eval_skip_high_card_cols: bool = not args.eval_disable_skip_high_card_cols

    wandb_project: Optional[str] = args.wandb_project
    wandb_entity: Optional[str] = args.wandb_entity
    dump_intermediates: bool = args.dump_intermediates

    ray.init(local_mode=local_mode)

    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    pl.seed_everything(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    project_dir = os.getcwd()
    cache_dir_root = os.path.join(project_dir, ".cache")

    # set params
    if do_train:
        # initialize
        loader = scardina.datasets.Loader(
            fact_threshold, join_sample_size_min, join_sample_size_max, device
        )
        db = loader.load(dataset_name, schema_strategy, cache_dir_root)

        table_names = db.tables.keys()

        if n_trials:  # train w/ hyperparam search
            params_dict = {}
            for table_name in table_names:
                params_dict[table_name] = {
                    "static": {
                        "device": device,
                        "dataset_name": dataset_name,
                        "model_type": model_type,
                        "schema_strategy": schema_strategy,
                        "cont_fanout": cont_fanout,
                        "table_name": table_name,
                        "n_epochs": n_epochs,  # behavior like max epoch
                        "fact_threshold": fact_threshold,
                        "smaller_emb": smaller_emb,
                        "id": timestamp,
                        "n_trials": n_trials,
                        "project_dir": project_dir,
                        "cache_dir_root": cache_dir_root,
                        "wandb_project": wandb_project,
                        "wandb_entity": wandb_entity,
                        "seed": random_seed,
                    },
                    "d_word": tune.choice([32, 64]),
                    "batch_size": 1024,
                    # "batch_size": 2,  # hack for tiny dataset
                    "lr": tune.loguniform(1e-4, 5e-3) if warmups is None else None,
                    "warmups": warmups,
                    "act_func_name": "gelu",
                }

                table = db.tables[table_name]

                input_bins: List[Tuple[str, int]] = (
                    [
                        (c.name, (c.dist_size if not c.is_fanout() else 1))
                        for c in table.scols
                    ]
                    if cont_fanout
                    else [(c.name, c.dist_size) for c in table.scols]
                )
                params_dict[table_name]["static"]["input_bins"] = input_bins

                if model_type == "trm":
                    if schema_strategy == "ur":
                        params_dict[table_name].update(
                            {
                                "n_blocks": tune.choice([1, 2]),
                                "d_ff": tune.choice([128, 256]),
                                "n_heads": tune.choice([1, 2]),
                            }
                        )
                    elif schema_strategy == "cin":
                        params_dict[table_name].update(
                            {
                                "n_blocks": tune.choice([1, 2]),
                                "d_ff": tune.choice([64, 96]),
                                "n_heads": tune.choice([1, 2]),
                            }
                        )
                    else:
                        raise NotImplementedError()
                elif model_type == "mlp":
                    if schema_strategy == "ur":
                        params_dict[table_name].update(
                            {
                                "d_ff": tune.choice([128, 256]),
                                "n_ff": tune.choice([4, 8]),
                            }
                        )
                    elif schema_strategy == "cin":
                        params_dict[table_name].update(
                            {"d_ff": tune.choice([64, 96]), "n_ff": tune.choice([2, 4])}
                        )
                    else:
                        raise NotImplementedError()
        else:  # train w/ specified hyperparams
            params_dict = {}
            for table_name in table_names:
                if params_dir is not None:
                    found = False
                    for params_file_path in glob.glob(
                        os.path.join(params_dir, "*.json")
                    ):
                        if table_name in params_file_path:
                            with open(params_file_path) as f:
                                params_dict[table_name] = json.load(f)["config"]
                                params_dict[table_name][
                                    "base_params_file_path"
                                ] = params_file_path
                                params_dict[table_name]["base_params_id"] = params_dict[
                                    table_name
                                ]["static"]["id"]
                                found = True
                                logging.info(
                                    f"Found config file for {table_name} ({params_file_path})"
                                )
                                break
                    if not found:
                        params_dict[table_name] = {}
                else:
                    params_dict[table_name] = {}

                def set_if_not_none(
                    dic: Dict[str, Any], key: str, val: Any, force=False
                ):
                    if (
                        val is None
                    ):  # NOTE: not support to set None even if force is True
                        if key not in dic:
                            dic[key] = None  # == val
                        return
                    if (key not in dic or dic[key] is None) or force:
                        if force:
                            logging.info(
                                f"Overriding config {table_name}[{key}] = {val}"
                            )
                        dic[key] = val

                params_dict_t = params_dict[table_name]
                set_if_not_none(params_dict_t, "d_word", d_word)
                set_if_not_none(params_dict_t, "batch_size", batch_size)
                set_if_not_none(params_dict_t, "lr", lr if warmups is None else None)
                params_dict_t["lr"] = (
                    params_dict_t["lr"] if "lr" in params_dict_t else None
                )
                set_if_not_none(params_dict_t, "warmups", warmups)
                params_dict_t["warmups"] = (
                    params_dict_t["warmups"] if "warmups" in params_dict_t else None
                )
                set_if_not_none(params_dict_t, "act_func_name", "gelu")

                if "static" not in params_dict[table_name]:
                    params_dict[table_name]["static"] = {}

                params_static = params_dict[table_name]["static"]
                set_if_not_none(params_static, "device", device, True)
                set_if_not_none(params_static, "dataset_name", dataset_name)
                set_if_not_none(params_static, "model_type", model_type)
                set_if_not_none(params_static, "schema_strategy", schema_strategy)
                set_if_not_none(params_static, "cont_fanout", cont_fanout)
                set_if_not_none(params_static, "table_name", table_name)
                set_if_not_none(params_static, "n_epochs", n_epochs, True)
                set_if_not_none(params_static, "fact_threshold", fact_threshold)
                set_if_not_none(params_static, "smaller_emb", smaller_emb)
                set_if_not_none(params_static, "id", timestamp, True)
                set_if_not_none(params_static, "project_dir", project_dir, True)
                set_if_not_none(params_static, "cache_dir_root", cache_dir_root, True)
                set_if_not_none(params_static, "wandb_project", wandb_project, True)
                set_if_not_none(params_static, "wandb_entity", wandb_entity, True)
                set_if_not_none(params_static, "seed", random_seed, True)

                table = db.tables[table_name]

                input_bins: List[Tuple[str, int]] = (
                    [
                        (c.name, (c.dist_size if not c.is_fanout() else 1))
                        for c in table.scols
                    ]
                    if cont_fanout
                    else [(c.name, c.dist_size) for c in table.scols]
                )
                set_if_not_none(params_static, "input_bins", input_bins)

                if model_type == "trm":
                    set_if_not_none(params_dict_t, "n_blocks", n_blocks, True)
                    set_if_not_none(params_dict_t, "d_ff", d_ff, True)
                    set_if_not_none(params_dict_t, "n_heads", n_heads, True)
                elif model_type == "mlp":
                    set_if_not_none(params_dict_t, "d_ff", d_ff, True)
                    set_if_not_none(params_dict_t, "n_ff", n_ff, True)

                # always None for avoiding hyperparam search
                params_static["n_trials"] = None
    elif do_eval:
        if schema_strategy == "cin":
            params_dict = {}
            for params_file_path in glob.glob(
                model_path.replace("{}", "*").replace(".pt", ".json")
            ):
                with open(params_file_path) as f:
                    verbose_params = json.load(f)
                    params = verbose_params["config"]
                table_name = params["static"]["table_name"]

                # update contextual params (different from train phase)
                params["static"]["device"] = device
                params["static"]["wandb_project"] = wandb_project
                params["static"]["wandb_entity"] = wandb_entity

                # add eval-specific params
                params["static"]["model_path"] = model_path.replace("{}", table_name)
                params["static"]["benchmark_name"] = benchmark_name
                params["static"]["eval_order"] = eval_order
                params["static"]["eval_sample_size"] = eval_sample_size
                params["static"]["eval_skip_high_card_cols"] = eval_skip_high_card_cols

                params_dict[table_name] = params
        elif schema_strategy == "ur":
            params_file_path = model_path.replace(".pt", ".json")

            with open(params_file_path) as f:
                verbose_params = json.load(f)
                params = verbose_params["config"]
            table_name = params["static"]["table_name"]

            # update contextual params (different from train phase)
            params["static"]["device"] = device
            params["static"]["wandb_project"] = wandb_project
            params["static"]["wandb_entity"] = wandb_entity

            # add eval-specific params
            params["static"]["model_path"] = model_path
            params["static"]["benchmark_name"] = benchmark_name
            params["static"]["eval_order"] = eval_order
            params["static"]["eval_sample_size"] = eval_sample_size
            params["static"]["eval_skip_high_card_cols"] = eval_skip_high_card_cols

            params_dict = {table_name: params}

        # initialize
        loader = scardina.datasets.Loader(
            list(params_dict.values())[0]["static"]["fact_threshold"], -1, -1, device
        )
        db = loader.load(dataset_name, schema_strategy, cache_dir_root)

    # run
    if do_train:
        train(params_dict, db)
    elif daemon_mode:
        raise NotImplementedError()
        # global DB, ESTIMATOR
        # DB = db
        # ESTIMATOR = eval_daemon(
        #     db, model, model_path, dataset_name, eval_n_samples, eval_order
        # )
        # API.run()
    elif do_eval:
        if not eval_all_intermediate_models:
            eval_batch(
                params_dict,
                db,
                log=log,
                dump_intermediates=dump_intermediates,
            )
        else:
            for _model_path in sorted(
                glob.glob(
                    os.path.join(
                        os.path.dirname(model_path), "intermediates", "*", "*.pt"
                    )
                ),
                key=lambda x: int(re.sub(r"\/.+$", "", re.sub("^.+step=", "", x))),
            ):
                for table_name, params in params_dict.items():
                    params["static"]["model_path"] = _model_path
                eval_batch(
                    params_dict,
                    db,
                    dump_intermediates=dump_intermediates,
                )
    # elif cache_data:
    #     for table_name in table_names:
    #         scardina.common.DBDataset(
    #             db, cont_fanout, table_name=table_name, cache_dir_root=cache_dir_root
    #         )


def create_model(
    params: Dict[str, Union[str, int, Dict[str, Union[str, int]]]]
) -> pl.LightningModule:
    input_bins: List[Tuple[str, int]] = params["static"]["input_bins"]
    model_type: str = params["static"]["model_type"]
    table_name = params["static"]["table_name"]

    if model_type == "trm":
        model = scardina.models.NARTransformer(
            params=params,
            table_name=table_name,
            input_bins=input_bins,
        )
    elif model_type == "mlp":
        model = scardina.models.NARMLP(
            params=params,
            table_name=table_name,
            input_bins=input_bins,
        )
    else:
        raise ValueError(f"Unexpected model type: {model_type}")

    return model.to(params["static"]["device"])


def train_table_trial(
    params: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
) -> None:
    data_module = ray.get(params.pop("data_module_ref"))
    project = params["static"]["wandb_project"]
    entity = params["static"]["wandb_entity"]
    online = (
        project is not None and project != "" and entity is not None and entity != ""
    )
    logger = pl_loggers.WandbLogger(
        name=f"{tune.get_trial_name()}_step",
        project=project,
        entity=entity,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
        mode=("online" if online else "offline"),
    )  # for per-step logging
    logger._experiment = wandb.init(
        **logger._wandb_init
    )  # manually init for parallel run

    table_name = params["static"]["table_name"]
    model = create_model(params)

    trainer = pl.Trainer(
        max_epochs=params["static"]["n_epochs"],
        callbacks=[
            TuneReportCheckpointCallback(
                metrics=[
                    "epoch",  # for ASHA scheduler (e.g., 1, 2, ..., n_epochs)
                    f"{table_name}/tra_loss",
                    f"{table_name}/val_loss",
                ],
                filename=f"{model.name}.pt",
                on="validation_end",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        gpus=1,
        num_sanity_val_steps=0,
        # related to logging
        # if set val_check_interval != 1.0,
        # it confuses ASHA scheduler w/ default time_attr (regard n steps as 1 epoch)
        # val_check_interval=0.25,
        log_every_n_steps=1,
        logger=logger,
    )
    trainer.fit(model, data_module)
    logger._experiment.finish()


def train_table(
    params: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
    data_module: pl.LightningDataModule,
    n_trials: Optional[int],  # `None` represents run w/ specified hyperparams
    loggerCallback: LoggerCallback = None,
) -> None:
    table_name = params["static"]["table_name"]

    if params["static"]["n_trials"] is not None:
        scheduler = ray.tune.schedulers.ASHAScheduler(
            time_attr="epoch",
            max_t=params["static"]["n_epochs"],
            # If set 1 w/ multiple validation steps, 1st validation of 1st ep
            # will be recognized as a representative metric of 1st ep.
            # As a result, the scheduler may cut off a trial at a too early step.
            # On the other hand, if set 2, run full 1ep (+alpha) at least.
            grace_period=max(params["static"]["n_epochs"] // 3, 2),
            reduction_factor=2,
        )
        search_alg = OptunaSearch(metric=f"{table_name}/val_loss", mode="min")
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)
    else:
        scheduler = None
        search_alg = None

    # NOTE: tune.with_parameters is buggy???
    # If pass data module via with_parameters,
    # tune runner includes data module into trainable func.
    # It makes trainable func huge and breaks serialization.
    data_module_ref = ray.put(data_module)
    params["data_module_ref"] = data_module_ref

    analysis = tune.run(
        train_table_trial,
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 12, "GPU": 1}]),
        local_dir=os.path.join(params["static"]["project_dir"], "runs"),
        num_samples=n_trials if n_trials is not None else 1,
        search_alg=search_alg,
        metric=f"{table_name}/val_loss",
        mode="min",
        config=params,
        scheduler=scheduler,
        # keep_checkpoints_num=5,
        checkpoint_score_attr=f"min-{table_name}/val_loss",
        reuse_actors=True,  # NOTE: no change w/ functional trainable
        callbacks=[loggerCallback],
    )

    print(f"best parameters: {analysis.best_config}")
    model_dir = os.path.join(
        params["static"]["project_dir"],
        "models",
        params["static"]["dataset_name"],
        params["static"]["model_type"] + "-" + params["static"]["schema_strategy"],
        params["static"]["id"],
    )
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = glob.glob(f"{analysis.best_checkpoint}*.pt")[0]
    best_model_name = os.path.splitext(os.path.basename(best_model_path))[0]
    shutil.copy2(best_model_path, model_dir)
    with open(os.path.join(model_dir, f"{best_model_name}.json"), "w") as f:
        json.dump(
            {
                **analysis.best_result,
                "best_logdir": analysis.best_logdir,
                "best_checkpointdir": analysis.best_checkpoint.local_path,
            },
            f,
            indent=2,
        )
    print(os.path.join(model_dir, f"{best_model_name}.pt"))

    # save all models for eval
    model_path_list = glob.glob(
        f"{analysis._checkpoints[0]['logdir']}/checkpoint*/*.pt"
    )
    for model_path in model_path_list:
        checkpoint_dir_name = re.findall(
            r"(?<=epoch=)\d+(?=-)", os.path.basename(os.path.dirname(model_path))[11:]
        )[0]
        intermediate_model_dir = os.path.join(
            model_dir,
            "intermediates",
            checkpoint_dir_name,  # epoch number
        )
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        os.makedirs(intermediate_model_dir, exist_ok=True)
        shutil.copy2(model_path, intermediate_model_dir)
        with open(os.path.join(intermediate_model_dir, f"{model_name}.json"), "w") as f:
            json.dump(
                {
                    **analysis.best_result,  # FIXME
                    # "best_logdir": analysis.best_logdir,
                    # "best_checkpointdir": analysis.best_checkpoint.local_path,
                },
                f,
                indent=2,
            )


def train(
    params_dict: Dict[str, Union[str, int]],
    db: scardina.common.DB,
):
    for table_name, params in params_dict.items():
        project = params["static"]["wandb_project"]
        entity = params["static"]["wandb_entity"]
        online = (
            project is not None
            and project != ""
            and entity is not None
            and entity != ""
        )
        loggerCallback = WandbLoggerCallback(
            project=project,
            entity=entity,
            api_key=(None if online else "dummy"),
            mode=("online" if online else "offline"),
        )  # NOTE: ray-tune LoggerCallback cannot report logs per-step

        data_module = scardina.common.DataModule(
            db.name,
            db.schema,
            db.vtable if db.vtable is not None else db.tables[table_name],
            params["batch_size"],
            params["static"]["fact_threshold"],
            cache_dir_root=params["static"]["cache_dir_root"],
            cont_fanout=params["static"]["cont_fanout"],
        )

        train_table(params, data_module, params["static"]["n_trials"], loggerCallback)


def eval_batch(
    params_dict: Dict[str, Union[str, int]],
    db: scardina.common.DB,
    log: logging.Logger,
    *,
    dump_intermediates=False,
):
    common_params = list(params_dict.values())[0]["static"]
    dataset_name = common_params["dataset_name"]
    model_type = common_params["model_type"]
    schema_strategy = common_params["schema_strategy"]
    benchmark_name = common_params["benchmark_name"]

    project = common_params["wandb_project"]
    entity = common_params["wandb_entity"]
    online = (
        project is not None and project != "" and entity is not None and entity != ""
    )
    wandb.init(
        name=f"{model_type}-{schema_strategy}/{dataset_name}-{benchmark_name}({common_params['id']})",
        project=project,
        entity=entity,
        config=params_dict,
        mode=("online" if online else "offline"),
    )

    if schema_strategy == "cin":
        model = {}
        for table_name, params in params_dict.items():
            if not os.path.exists(params["static"]["model_path"]):
                raise ValueError(f"Model not found: {params['static']['model_path']}")
            m = create_model(params)
            m.load_state_dict(torch.load(params["static"]["model_path"])["state_dict"])
            m.eval()
            model[table_name] = m
    elif schema_strategy == "ur":
        assert len(params_dict) == 1
        params = list(params_dict.values())[0]
        if not os.path.exists(params["static"]["model_path"]):
            raise ValueError(f"Model not found: {params['static']['model_path']}")
        model = create_model(params)
        model.load_state_dict(torch.load(params["static"]["model_path"])["state_dict"])
        model.eval()
    else:
        raise ValueError(f"Unknown relation type: {schema_strategy}")

    master_dataset_name = db.schema.master_name
    benchmark = pa.csv.read_csv(
        os.path.join("benchmarks", master_dataset_name, f"{benchmark_name}.csv")
    ).to_pandas()
    true_cards = benchmark["true_cardinality"].values
    conds_list = [scardina.parse.parse_to_conds(sql) for sql in benchmark["sql"].values]
    # TODO: 別moudleに切り出す
    queries = []
    for i, conds in enumerate(conds_list):
        cols = []
        ops = []
        vals = []
        tbls = set()
        joins = []

        # TMP
        if len(conds) == 0:
            parsed = sp.parse(benchmark["sql"].values[i])
            from_clause = False
            for token in parsed[0].tokens:
                if token.is_whitespace:
                    continue
                if from_clause:
                    tbls.add(token.normalized)
                    break
                if token.normalized == "FROM":
                    from_clause = True

        for cond in conds:
            is_join = cond[3]
            join = []
            col_name = ""
            for t in cond[1].tokens:
                if t.ttype == sp.tokens.Name:
                    if col_name != "" and is_join:
                        # <already parsed>.<HERE> and is_join
                        join[-1].append(
                            t.value
                            if not db.schema.force_lower_case
                            else t.value.lower()
                        )
                    col_name = col_name + t.value
                elif t.ttype == sp.tokens.Literal.String.Symbol:
                    if t.value.startswith('"'):
                        t.value = t.value[1:]
                    if t.value.endswith('"'):
                        t.value = t.value[:-1]
                    col_name = col_name + t.value
                elif t.ttype == sp.tokens.Punctuation:
                    tbls.add(col_name)
                    if is_join:
                        join.append([col_name])
                    col_name = col_name + "."
                else:
                    assert False, f"col_name contains invalid tokens: {cond[1]}"
            val = ""
            if not is_join:

                def _parse_val(t):
                    if t is None:
                        return None
                    elif isinstance(t, list):
                        return [_parse_val(x) for x in t]
                    elif t.ttype == sp.tokens.Literal.Number.Integer:
                        return int(t.value)
                    elif t.ttype == sp.tokens.Literal.Number.Float:
                        return float(t.value)
                    elif t.ttype == sp.tokens.Literal.String.Single:
                        if t.value.startswith("'"):
                            t.value = t.value[1:]
                        if t.value.endswith("'"):
                            t.value = t.value[:-1]

                        # NOTE: super heuristic! use schema.type_casts
                        if "date" in col_name.lower():
                            return np.datetime64(t.value)
                        else:
                            return str(t.value)
                    elif t.ttype == sp.tokens.Keyword and t.value.upper() == "NULL":
                        return None
                    else:
                        assert False, f"Not supported type: {t}"

                val = _parse_val(cond[2])
            else:
                for t in cond[2].tokens:
                    if t.ttype == sp.tokens.Name:
                        if val != "":
                            join[-1].append(
                                t.value
                                if not db.schema.force_lower_case
                                else t.value.lower()
                            )
                        val = val + t.value
                    elif t.ttype == sp.tokens.Punctuation:
                        tbls.add(val)
                        join.append([val])
                        val = val + "."
                    else:
                        assert False, f"val contains invalid tokens: {cond[2]}"

            if not is_join:
                # TMP
                if dataset_name != "dmv":
                    cols.append(col_name.lower())
                else:
                    cols.append(col_name)
                ops.append(cond[0].value.upper())
                vals.append(val)
                # if col_name in col_name_to_col:  # TMP
                #     cols.append(col_name)
                #     ops.append(cond[0].value.upper())
                #     vals.append(val)
                # else:
                #     print(f"missing {col_name}")
            else:
                joins.append(join)

        queries.append(
            (
                np.array(cols),
                np.array(ops, dtype=np.str),
                np.array(vals, dtype=np.object),
                tbls,
                joins,
                benchmark["sql"].values[i],
            )
        )

    assert len(queries) == len(true_cards)
    n_queries = len(queries)

    if not isinstance(model, dict):
        estimator = scardina.estimators.ProgressiveSamplingUR(
            model,
            db,
            params_dict,
            dump_intermediates=dump_intermediates,
            log=log,
        )
    else:
        estimator = scardina.estimators.ProgressiveSamplingCIN(
            model,
            db,
            params_dict,
            log=log,
        )
    for i in range(n_queries):
        query(
            estimator,
            queries[i],
            true_cards[i],
            log,
            i=i,
        )
    result = {
        "estimator": [estimator.name] * n_queries,
        "q_err": estimator.errs,
        "estimated_cardinality": estimator.est_cards,
        "true_cardinality": estimator.true_cards,
        "elapsed_time_ms": estimator.elapsed_time_ms,
    }
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_dir_path = os.path.join(
        "results",
        dataset_name,
        benchmark_name,
        common_params["id"],
        timestamp,
    )
    result_file_name = (
        f"{dataset_name}_{model_type}_{schema_strategy}"
        + f"_{common_params['id']}"
        + f"_{timestamp}.csv"
    )
    result_file_path = os.path.join(result_dir_path, result_file_name)
    result = pd.DataFrame(result)
    xs = (
        result["q_err"]
        .quantile(q=[0.5, 0.9, 0.95, 0.99, 1.0], interpolation="nearest")
        .tolist()
    )
    xs.append(result["elapsed_time_ms"].mean())
    print("q-err: [" + ", ".join([f"{x:.3f}" for x in xs]) + "]")
    os.makedirs(result_dir_path, exist_ok=True)
    result.to_csv(result_file_path, index=False)
    with open(result_file_path.replace(".csv", ".json"), "w") as f:
        json.dump(params_dict, f, indent=2)
    print(result_file_name)

    ppc_eval = True
    if ppc_eval and "name" in benchmark:
        # save in pickle for ppc eval
        result_pickle_file_path = result_file_path.replace(".csv", ".pkl")
        preds = {}
        for i, row in benchmark.iterrows():
            if row["name"] not in preds:
                preds[row["name"]] = {}
            preds[row["name"]][row["node"]] = estimator.est_cards[i]
        with open(result_pickle_file_path, "wb") as f:
            pickle.dump(preds, f)

        # TMP
        db_config = {
            "db_name": master_dataset_name,
            "user": "ceb",
            "pwd": "password",
            "db_host": "card-db",
            "port": "5432",
        }

        eval_ppc(
            result_dir_path,
            result_file_path,
            result_pickle_file_path,
            master_dataset_name,
            benchmark_name,
            db_config,
        )


def eval_ppc(
    result_dir_path: str,
    result_file_path: str,
    result_pickle_file_path: str,
    master_dataset_name: str,
    benchmark_name: str,
    db_config: Dict[str, str],
):
    assert benchmark_name.endswith("_subqueries")
    benchmark_name = benchmark_name[:-11]
    perr_result_file_path = result_file_path.replace(".csv", "_perr.csv")
    ceb_result_dir_path = os.path.join(result_dir_path, "ppc")
    qreps_dir_path = os.path.join(
        "resources", "ceb", "queries", master_dataset_name, benchmark_name
    )
    qreps_glob = os.path.join(qreps_dir_path, "*.pkl")
    os.makedirs(ceb_result_dir_path, exist_ok=True)
    if not os.path.exists(qreps_dir_path):
        print("warn: Skipped ppc eval (qreps not found)")
        return

    # load pickle
    query_file_paths = sorted(
        list(glob.glob(qreps_glob)), key=lambda p: int(p.split("/")[-1][:-4])
    )
    qreps = []
    for query_file_path in tqdm.tqdm(query_file_paths):
        qrep = query_representation.query.load_qrep(query_file_path)
        qrep["name"] = os.path.basename(query_file_path)
        qreps.append(qrep)

    eval_fns = {
        "ppc": evaluation.eval_fns.get_eval_fn("ppc"),
        "perr": evaluation.eval_fns.get_eval_fn("perr"),
    }
    algs = {
        "true": cardinality_estimation.algs.TrueCardinalities(),
        "est": cardinality_estimation.algs.SavedPreds(
            file_path=result_pickle_file_path
        ),
    }
    res = {}
    for alg_name, alg in algs.items():
        ests = alg.test(qreps)
        for eval_name, eval_fn in eval_fns.items():
            ceb_result_file_path = os.path.join(
                ceb_result_dir_path, f"{eval_fn.__class__.__name__}_{alg_name}card.csv"
            )
            if os.path.exists(ceb_result_file_path):
                print("found cache")
                res[(alg_name, eval_name)] = (ceb_result_file_path, None)
            else:
                res[(alg_name, eval_name)] = (
                    ceb_result_file_path,
                    eval_fn.eval(
                        qreps,
                        ests,
                        args={},
                        samples_type="test",
                        result_dir=ceb_result_dir_path,
                        **db_config,
                        num_processes=4,
                        alg_name=alg_name,
                    ),
                )
                shutil.move(
                    os.path.join(
                        ceb_result_dir_path, f"{eval_fn.__class__.__name__}.csv"
                    ),
                    ceb_result_file_path,
                )
    ppc_truecard = pd.read_csv(res[("true", "ppc")][0]).sort_values("id")
    ppc_estcard = pd.read_csv(res[("est", "ppc")][0]).sort_values("id")
    p_err = pd.read_csv(res[("est", "perr")][0]).sort_values("id")
    result_df = ppc_truecard.rename(
        columns={
            "id": "query_no",
            "cost": "cost_truecard",
            "join_order": "join_order_truecard",
        },
    )
    # e.g., (t (mc cn)) into ["t",["mc","cn"]] (JSON format)
    result_df["join_order_truecard"] = (
        result_df["join_order_truecard"]
        .str[7:]
        .str.replace("(", "[", regex=False)
        .str.replace(")", "]", regex=False)
        .str.replace(" ", ",", regex=False)
        .str.replace(r"([\w_]+)", lambda x: f'"{x.group(0)}"', regex=True)
    )
    result_df["cost_estcard"] = ppc_estcard["cost"]
    result_df["join_order_estcard"] = ppc_estcard["join_order"]
    result_df["join_order_estcard"] = (
        result_df["join_order_estcard"]
        .str[7:]
        .str.replace("(", "[", regex=False)
        .str.replace(")", "]", regex=False)
        .str.replace(" ", ",", regex=False)
        .str.replace(r"([\w_]+)", lambda x: f'"{x.group(0)}"', regex=True)
    )
    result_df["p_err"] = p_err["errors"]
    result_df = result_df.reindex(
        columns=[
            "query_no",
            "p_err",
            "cost_estcard",
            "cost_truecard",
            "join_order_estcard",
            "join_order_truecard",
        ]
    )

    result_df.to_csv(perr_result_file_path, index=False)
    p_err_percentiles = result_df["p_err"].quantile(
        [0.5, 0.9, 0.95, 0.99, 1.0], interpolation="nearest"
    )
    print("p-err: [" + ", ".join([f"{e:.3f}" for e in p_err_percentiles]) + "]")


if __name__ == "__main__":
    main()
