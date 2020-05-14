#!/bin/bash

# Run a decoupled smoothing method against all data variations.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DATA_DIR="${THIS_DIR}/data"

# An identifier to differentiate the output of this script/experiment from other scripts.
readonly RUN_ID='run-all-decoupled-smoothing'

function generate_data() {
  random_seed=$1
  data_name=$2

  echo "Generating data with seed ${random_seed} and data ${data_name}"

  printf -v seed_nm "%04d" $random_seed
  local logPath="${BASE_DATA_DIR}/${data_name}/01pct/${seed_nm}rand/data_log.json"

  echo "${logPath}"

  if [[ -e "${logPath}" ]]; then
    echo "Output data already exists, skipping data generation"
  else
    python3 write_psl_data.py  --seed ${random_seed} --data ${data_name}.mat
  fi
}

function main() {
  # help text
  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <method cli_dir> <data name> <random seed> ..."
    exit 1
  fi

  method=$1
  data_name=$2
  random_seed=$3

  trap exit SIGINT

  # running all methods for all seeds
  if [ $method == "all" ]; then
    for rand_sd in 837 2841 4293 6305 6746 9056 9241 9547; do
      # generate data for the random seed
      generate_data "${rand_sd}" "${data_name}"

      # find all the results for each percentage
      echo "Running ${method} for all percentages"
      for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95 99; do
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "cli_one_hop/"
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "cli_two_hop/"
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "cli_decoupled_smoothing_mod/"
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "cli_decoupled_smoothing_prior/"
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "cli_decoupled_smoothing_partial/"
      done
    done
    return 0
  # running all percentages for one random seed and one model
  else
    generate_data "${random_seed}" "${data_name}"

    echo "Running ${method} for all percentages"
    for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95 99; do
      ./run_method.sh "${data_name}" "${random_seed}" "${pct_lbl}" "${method}"
    done
  fi
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
