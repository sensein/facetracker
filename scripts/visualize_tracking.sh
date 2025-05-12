#!/bin/bash

#SBATCH --job-name=visualize_tracking
#SBATCH --output=../slurm_logs/visualize_tracking_%j.out
#SBATCH --error=../slurm_logs/visualize_tracking_%j.err
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4     # Adjust as needed
#SBATCH --mem=4G             # Adjust as needed
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --time=01:00:00       # Adjust as needed (e.g., 1 hour)

# --- User Configuration ---
# Path to the Apptainer SIF file
SIF_PATH="/home/yibei/images/facetracker_0-2-1.sif"

# Host path for the project directory.
# This directory will be bound to /opt/facetracker inside the container.
# Outputs will be placed relative to this directory if CWD is this directory.
PROJECT_HOST_DIR="/home/yibei/facetracker"

# Host path to the input video file.
# Its directory will be bound to /input_videos (read-only) inside the container.
INPUT_VIDEO_HOST_PATH="/orcd/scratch/bcs/001/yibei/friends/data/mkv2mp4/friends_s01e01a.mp4"

# Host path to the face landmarker model file (.task).
# Its directory will be bound to /model_assets (read-only) inside the container.
FACE_MODEL_HOST_PATH="/orcd/data/satra/002/models/mediapipe/face_landmarker.task"

# Subdirectory (relative to current working directory, assumed to be PROJECT_HOST_DIR) for the output video
OUTPUT_VIDEO_SUBDIR="facetracker_pipeline_output"

# Name of the output video file
OUTPUT_VIDEO_NAME="video_tracking_retinaface.mp4"
# --- End User Configuration ---

# --- Path Derivations for Apptainer ---
# Container bind targets
PROJECT_CONTAINER_DIR="/opt/facetracker"
INPUT_VIDEO_CONTAINER_BIND_DIR="/input_videos"
MODEL_CONTAINER_BIND_DIR="/model_assets"

# Derive source directories and filenames for binds and script arguments
INPUT_VIDEO_HOST_BASEDIR=$(dirname "${INPUT_VIDEO_HOST_PATH}")
INPUT_VIDEO_FILENAME=$(basename "${INPUT_VIDEO_HOST_PATH}")
MODEL_HOST_BASEDIR=$(dirname "${FACE_MODEL_HOST_PATH}")
MODEL_FILENAME=$(basename "${FACE_MODEL_HOST_PATH}")

# Path to the python script inside the container
PYTHON_SCRIPT_IN_CONTAINER="${PROJECT_CONTAINER_DIR}/scripts/visualize_tracking.py"

# Paths for arguments to the python script (inside container)
VIDEO_ARG_PATH_IN_CONTAINER="${INPUT_VIDEO_CONTAINER_BIND_DIR}/${INPUT_VIDEO_FILENAME}"
MODEL_ARG_PATH_IN_CONTAINER="${MODEL_CONTAINER_BIND_DIR}/${MODEL_FILENAME}"
OUTPUT_ARG_PATH_IN_CONTAINER="${PROJECT_CONTAINER_DIR}/${OUTPUT_VIDEO_SUBDIR}/${OUTPUT_VIDEO_NAME}"

# Full path for output on the host (assuming CWD is PROJECT_HOST_DIR for relative OUTPUT_VIDEO_SUBDIR)
FULL_OUTPUT_VIDEO_PATH_ON_HOST="${OUTPUT_VIDEO_SUBDIR}/${OUTPUT_VIDEO_NAME}"
# --- End Path Derivations ---

# Create output directories if they don't exist
# slurm_logs is relative to where sbatch is run or script location if not specified otherwise
mkdir -p ../slurm_logs
# OUTPUT_VIDEO_SUBDIR is created relative to CWD (assumed to be PROJECT_HOST_DIR)
mkdir -p "${OUTPUT_VIDEO_SUBDIR}"

# Echo configuration
echo "Starting visualization script..."
echo "SIF File: ${SIF_PATH}"
echo "Project Host Directory: ${PROJECT_HOST_DIR}"
echo "Input Video (Host): ${INPUT_VIDEO_HOST_PATH}"
echo "Face Model (Host): ${FACE_MODEL_HOST_PATH}"
echo "Output Video (Host, relative to CWD): ${FULL_OUTPUT_VIDEO_PATH_ON_HOST}"
echo "---"
echo "Python Script (Container): ${PYTHON_SCRIPT_IN_CONTAINER}"
echo "Video Path Arg (Container): ${VIDEO_ARG_PATH_IN_CONTAINER}"
echo "Model Path Arg (Container): ${MODEL_ARG_PATH_IN_CONTAINER}"
echo "Output Path Arg (Container): ${OUTPUT_ARG_PATH_IN_CONTAINER}"
echo "---"

# Load Apptainer module (specific to HPC environment)
module load apptainer/1.1.9

# Run the Python script using Apptainer
apptainer exec --nv \
    --bind "${PROJECT_HOST_DIR}:${PROJECT_CONTAINER_DIR}" \
    --bind "${INPUT_VIDEO_HOST_BASEDIR}:${INPUT_VIDEO_CONTAINER_BIND_DIR}:ro" \
    --bind "${MODEL_HOST_BASEDIR}:${MODEL_CONTAINER_BIND_DIR}:ro" \
    "${SIF_PATH}" \
    python3 "${PYTHON_SCRIPT_IN_CONTAINER}" \
        --video_path "${VIDEO_ARG_PATH_IN_CONTAINER}" \
        --output_video_path "${OUTPUT_ARG_PATH_IN_CONTAINER}" \
        --face_landmarker_model_path "${MODEL_ARG_PATH_IN_CONTAINER}"

echo "Script finished." 