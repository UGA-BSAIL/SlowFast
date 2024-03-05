#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on HPG.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH --partition=gpu
#SBATCH -J self_supervised_video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=40gb
#SBATCH --account=lift-phenomics
#SBATCH --qos=lift-phenomics
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=self_supervised_video_train.%j.out    # Standard output log
#SBATCH --error=self_supervised_video_train.%j.err     # Standard error log

set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/blue/lift-phenomics/$(whoami)/job_scratch/"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/blue/lift-phenomics/$(whoami)/slowfast/"
# Local copy of the dataset.
LOCAL_DATA_DIR="${SLURM_TMPDIR}/data/"

function prepare_environment() {
  # Copy the entire dataset to local scratch.
  echo "Copying dataset..."
#  cp "${LARGE_FILES_DIR}/data/mars_multi_camera.tar" "${LOCAL_DATA_DIR}"
#  tar -xf "${LOCAL_DATA_DIR}/mars_multi_camera.tar" -C "${LOCAL_DATA_DIR}"
#  mv "${LOCAL_DATA_DIR}/scratch/local/"*/* "${LOCAL_DATA_DIR}/"

  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${job_dir}/"

  # Link to the input data directory and venv.
  rm -rf "${job_dir}/data"
  ln -s "${LOCAL_DATA_DIR}" "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/venv" "${job_dir}/.venv"

  # Set the working directory correctly.
  cd "${job_dir}"
}

# Prepare the environment.
prepare_environment

source .venv/bin/activate
python -m tools.run_net --cfg configs/contrastive_ssl/SimCLR_SlowR50_8x8_Cotton_spatial_only.yaml
