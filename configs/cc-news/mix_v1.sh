#! /usr/bin/env bash


# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"


# run years between 2016 and 2024
for year in {2016..2024}; do
    # run months between 1 and 12
    for month in {1..12}; do
        # skip months after 7 if year is 2024
        if [ $year -eq 2024 ] && [ $month -gt 7 ]; then
            continue
        fi

        # skip months before 8 if year is 2016
        if [ $year -eq 2016 ] && [ $month -lt 8 ]; then
            continue
        fi

        # rename month to 2 digits
        month=$(printf "%02d" $month)

        # run deduplication
        echo "Mixing ${year}-${month}"

        export MIX_MONTH=${month}
        export MIX_YEAR=${year}

        set -ex

        dolma -c ${SCRIPT_DIR}/mix_v1.yaml mix

        set +ex
    done
done
