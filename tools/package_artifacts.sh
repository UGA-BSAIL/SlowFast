#!/bin/bash

# Helper script that packages artifacts from a training run.

#SBATCH --partition=hpg-default
#SBATCH -J cotton_mot_model_package_artifacts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --mem=2gb
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --account=lift-phenomics
#SBATCH --qos=lift-phenomics
#SBATCH --output=ssl_model_package_artifacts.out    # Standard output log
#SBATCH --error=ssl_model_package_artifacts.err     # Standard error log

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 JOB_ID"
  exit 1
fi

# The job ID to collect artifacts from.
JOB_ID=$1
# Split the job ID on the dot.
job_dir="/blue/lift-phenomics/$(whoami)/job_scratch/job_${JOB_ID}"

function package_artifacts() {
  mkdir artifacts

  # Grab the job output.
  zip artifacts/output.zip *_video_train."${JOB_ID}".*

  # Remove some checkpoints to save space.
  rm -f "${job_dir}/checkpoints/ssl_checkpoint_epoch_000*"
  # Grab the models
  zip -r artifacts/models.zip "${job_dir}/checkpoints/"

  # Grab the logs.
  zip -r artifacts/logs.zip "${job_dir}/logs/"
}

function clean_artifacts() {
  # Remove old job data.
  rm -rf "${job_dir}"
  # Remove old job output.
  rm *_model_train."${JOB_ID}".*
}

package_artifacts
clean_artifacts
