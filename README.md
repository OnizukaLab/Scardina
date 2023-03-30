# Scardina: Scalable Join Cardinality Estimatior

## Prerequisites
All experiments can be run in a docker container.

* Docker
* GPU/cuda environment (for Training)


## Getting Started
### Setup
Dependencies are automatically installed while building a docker image.

```bash
# on host
git clone https://github.com/OnizukaLab/Scardina.git
cd Scardina
docker build -t scardina .
docker run --rm --gpus all -v `pwd`:/workspaces/scardina -it scardina bash

# in container
poetry shell

# in poetry env in container
./scripts/dowload_imdb.sh
```


### Examples
#### Training
Choose hyperparameter search by optuna or manually specified parameters.
```bash
# train w/ hyperparameter search
python scardina/run.py --train -d=imdb -t=mlp --n-trials=10 -e=20

# train w/o hyperparameter search
python scardina/run.py --train -d=imdb -t=mlp -e=20 --d-word=64 --d-ff=256 --n-ff=4 --lr=5e-4
```


#### Evaluation
```bash
# evaluation
# Note: When default (-s=cin), model path should be like:
#       "models/imdb/mlp-cin/yyyyMMddHHmmss/nar-mlp-imdb-{}-yyyyMMddHHmmss.pt".
#       "{}" is literally "{}", a placeholder string to specify multiple models
python scardina/run.py --eval -d=imdb -b=job-light -t=mlp -m={path/to/model.pt}
```
You can find results in `results/<benchmark_name>` after trial.


### Options
#### Common Options
* `-d/--dataset`: Dataset name
* `-t/--model-type`: Internal model type (`mlp` for MLP or `trm` for Transformer)
* `-s/--schema-strategy`: Internal subschema type (`cin` for Closed In-neighborhood Partitioning (Scardina) or `ur` for Universal Relation)
* `--seed`: Random seed (Default: `1234`)
* `--n-blocks`: The number of blocks (for Transformer)
* `--n-heads`: The number of heads (for Transformer)
* `--d-word`: Embedding dimension
* `--d-ff`: Width of feedforward networks
* `--n-ff`: The number of feedforward networks (for MLP)
* `--fact-threshold`: [Column factorization](https://speakerdeck.com/zongheng/neurocard-one-cardinality-estimator-for-all-tables?slide=27) threshold (Default: `2000`)

#### Options for Training
* `-e/--epochs`: Training epoch
* `--batch-size`: Batch size (Default: `1024`)

(w/ hyperparameter search)
* `--n-trials`: The number of trials for hyperparameter search

(w/ specified parameters)
* `--lr`: Learning rate
* `--warmups`: Warm-up epoch (for Transformer) (`lr` and `warmups` are exclusive)

#### Options for Evaluation
* `-m/--model`: Path to model
* `-b/--benchmark`: Benchmark name
* `--eval-sample-size`: Sample size for evaluation

#### Choices
* Datasets
    * IMDb
        * `imdb`: (almost) All data of IMDb
        * `imdb-job-light`: Subset of IMDb for JOB-light benchmark
* Benchmarks
    * IMDb
        * `job-light`: Real-world 70 queries
        * `job-m`: Real-world 113 queries
        * `job-light_subqueries`: Real-world 70 queries for evaluating P-Error (Need [DB](https://github.com/OnizukaLab/CEB))
        * `job-m_subqueries`: Real-world 113 queries for evaluating P-Error (Need [DB](https://github.com/OnizukaLab/CEB))
* Models
    * `mlp`: MLP-based denoising autoencoder
    * `trm`: Transformer-based denoising autoencoder


## Reference
```bib
@article{scardina,
    author = {Ito, Ryuichi and Sasaki, Yuya and Xiao, Chuan and Onizuka, Makoto},
    title = {{Scardina: Scalable Join Cardinality Estimation by Multiple Density Estimators}},
    journal = {{arXiv preprint arXiv:XXXX.XXXXX}},
    year = {2023}
}
```


## Acknowledgement
Some source codes are based on [naru](https://github.com/naru-project/naru)/[neurocard](https://github.com/neurocard/neurocard)
