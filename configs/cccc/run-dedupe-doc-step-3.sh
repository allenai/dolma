#!/bin/bash

set -ex

dolma -c configs/cccc/dedupe-doc/CC-MAIN-2016-26.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2016-30.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2017-30.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2018-09.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2019-43.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2019-51.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2020-10.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2020-24.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2020-40.yaml dedupe
dolma -c configs/cccc/dedupe-doc/CC-MAIN-2023-40.yaml dedupe