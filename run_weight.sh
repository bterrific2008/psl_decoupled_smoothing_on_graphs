#!/bin/bash

# Run a decoupled smoothing method against all data variations.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DATA_DIR="${THIS_DIR}/data"

# An identifier to differentiate the output of this script/experiment from other scripts.
# readonly RUN_ID='run-all-weight-decoupled-smoothing'

function generate_data() {
  random_seed=$1
  data_name=$2
  train_test=$3

  printf -v seed_nm "%04d" $random_seed
  local logPath="${BASE_DATA_DIR}/${train_test}/${data_name}/01pct/${seed_nm}rand/data_log.json"

  if [[ -e "${logPath}" ]]; then
    echo "Output data already exists, skipping data generation"
  elif [ "$train_test" = learn ]; then
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 write_psl_data.py --seed ${random_seed} --data ${data_name}.mat --learn
  else
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 write_psl_data.py --seed ${random_seed} --data ${data_name}.mat
  fi
}

function main() {
  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <method cli_dir> <data name>  ..."
    exit 1
  fi

  method=$1
  data_name=$2

  trap exit SIGINT

  if [ $method == "all" ]; then
    # learn the learn data
    generate_data 4212 "${data_name}" "learn"

    # eval the data
    for pct_lbl in 01 10 30 50 80; do
      for sub_method in cli_decoupled_smoothing_pref_homophily/; do
        echo "learn: Random 4212 | PCT ${pct_lbl} | method ${sub_method}"
        ./run_method.sh "${data_name}" "4212" "${pct_lbl}" "learn" "${sub_method}"

        for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
          echo "eval: Random ${rand_sd} | PCT ${pct_lbl} | method ${sub_method}"
          generate_data "${rand_sd}" "${data_name}" "eval"
          ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "eval" "${sub_method}"
        done
      done
    done

    return 0
  else
    generate_data 4212 "${data_name}" "learn"

    echo "Running ${method} for all percentages"
    for pct_lbl in 01 20 50 80 95; do
      echo "learn: Random 4212 | PCT ${pct_lbl} | method ${method}"
      ./run_method.sh "${data_name}" "4212" "${pct_lbl}" "learn" "${method}"

      for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
        echo "eval: Random ${rand_sd} | PCT ${pct_lbl} | method ${method}"
        generate_data "${rand_sd}" "${data_name}" "eval"
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "eval" "${method}"
      done
    done
  fi
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
