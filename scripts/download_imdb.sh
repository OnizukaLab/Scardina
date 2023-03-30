#!/bin/bash

mkdir -p datasets/imdb
curl -L http://homepages.cwi.nl/~boncz/job/imdb.tgz | tar xz -C datasets/imdb
python scripts/add_header_imdb.py

# If want to manually generate join samples, see instruction in the README
# curl -L https://github.com/OnizukaLab/nar-cardest/releases/download/v0.1.0/imdb.csv.gz | gzip -d > datasets/imdb.csv
# curl -L https://github.com/OnizukaLab/nar-cardest/releases/download/v0.1.0/imdb-job-light.csv.gz | gzip -d > datasets/imdb-job-light.csv
