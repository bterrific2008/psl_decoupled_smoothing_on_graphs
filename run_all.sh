#!/bin/bash

# Run a single split of all the specified psl-examples.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_OUT_DIR="${THIS_DIR}/results"
readonly BASE_DATA_DIR="${THIS_DIR}/data"

readonly ADDITIONAL_PSL_OPTIONS='-D log4j.threshold=DEBUG'

# An identifier to differentiate the output of this script/experiment from other scripts.
readonly RUN_ID='single-split'

function run_psl() {
    local cliDir=$1
    local outDir=$2
    local extraOptions=$3

    mkdir -p "${outDir}"

    local outPath="${outDir}/out.txt"
    local errPath="${outDir}/out.err"
    local timePath="${outDir}/time.txt"

    if [[ -e "${outPath}" ]]; then
        echo "Output file already exists, skipping: ${outPath}"
        return 0
    fi

    pushd . > /dev/null
        cd "${cliDir}"

        # Run PSL.
        /usr/bin/time -v --output="${timePath}" ./run.sh ${extraOptions} > "${outPath}" 2> "${errPath}"

        # Copy any artifacts into the output directory.
        cp -r inferred-predicates "${outDir}/"
        cp *.data "${outDir}/"
        cp *.psl "${outDir}/"
    popd > /dev/null
}

function run_example() {
    local exampleDir=$1

    local exampleName=`basename "${exampleDir}"`
    local cliDir="$exampleDir/cli"
    local outDir="${BASE_OUT_DIR}/${RUN_ID}/${exampleName}"
    local options="${ADDITIONAL_PSL_OPTIONS}"

    echo "Running ${exampleName} -- ${RUN_ID}."

    run_psl "${cliDir}" "${outDir}" "${options}"
}

function generate_data() {
    random_seed=$1
    data_name=$2

    python3 write_psl_data.py  --seed ${random_seed} --data ${data_name}.mat
}

function main() {
    if [[ $# -eq 0 ]]; then
        echo "USAGE: $0 <one | two | decouple> <data name> <random seed> ..."
        exit 1
    fi

    method=$1
    data_name=$2
    random_seed=$3

    trap exit SIGINT

    echo "Generating data with seed ${random_seed} and data ${data_name}"

    generate_data "${random_seed}" "${data_name}"

    method_dir=""
    if [[ "$method" == "one" ]]; then
      method_dir="cli_one_hop/"
    elif [[ "$method" == "two" ]]; then
      method_dir="cli_two_hop/"
    elif [[ "$method" == "decouple" ]]; then
      method_dir="cli_decoupled_smoothing/"
    fi

    echo "Running ${method_dir} for all percentages"

    ./run_method.sh "${data_name}" "${random_seed}" "01" "${method_dir}"

    # for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95 99; do
    #   run_method "${data_name}" "${random_seed}" "${pct_lbl}" "${method_dir}"
    # done

}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"