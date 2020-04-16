#!/bin/bash

# Run a decoupled smoothing method against all data variations.

# An identifier to differentiate the output of this script/experiment from other scripts.
readonly RUN_ID='run-all-decoupled-smoothing'

function generate_data() {
    random_seed=$1
    data_name=$2

    python3 write_psl_data.py  --seed ${random_seed} --data ${data_name}.mat
}

function main() {
    if [[ $# -eq 0 ]]; then
        echo "USAGE: $0 <method cli_dir> <data name> <random seed> ..."
        exit 1
    fi

    method=$1
    data_name=$2
    random_seed=$3

    trap exit SIGINT

    echo "Generating data with seed ${random_seed} and data ${data_name}"

    generate_data "${random_seed}" "${data_name}"

    echo "Running ${method} for all percentages"

    ./run_method.sh "${data_name}" "${random_seed}" "01" "${method_dir}"

    for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95 99; do
      run_method "${data_name}" "${random_seed}" "${pct_lbl}" "${method}"
    done

}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
