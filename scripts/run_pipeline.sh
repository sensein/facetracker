#!/bin/bash

#SBATCH --job-name=run_pipeline
#SBATCH --output=../slurm_logs/run_pipeline_%A_%a.out
#SBATCH --error=../slurm_logs/run_pipeline_%A_%a.err
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4     # Adjust as needed
#SBATCH --mem=4G             # Adjust as needed
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --time=01:30:00       # Adjust as needed (e.g., 1 hour)
#SBATCH --array=0-2         # Array of 292 jobs (0 to 291)

# --- User Configuration ---
# Path to the Apptainer SIF file
SIF_PATH="/home/yibei/images/facetracker_0-2-1.sif"

# Host path for the project directory.
# This directory will be bound to /opt/facetracker inside the container.
PROJECT_HOST_DIR="/home/yibei/facetracker"

# Host path to the input video directory
INPUT_VIDEO_HOST_DIR="/orcd/scratch/bcs/001/yibei/friends/data/mkv2mp4"

# Host path to the output directory
OUTPUT_HOST_DIR="/orcd/scratch/bcs/001/yibei/friends/output/face_event"

# Host paths to the model files
FACE_MODEL_HOST_PATH="/orcd/data/satra/002/models/mediapipe/face_landmarker.task"
POSE_MODEL_HOST_PATH="/orcd/data/satra/002/models/mediapipe/pose_landmarker_full.task"

# Get the video filename from the array index
# Create an array of all video files
mapfile -t VIDEO_FILES < <(ls -1 "${INPUT_VIDEO_HOST_DIR}"/*.mp4)

# Check if the array index is valid
if [ ${SLURM_ARRAY_TASK_ID} -ge ${#VIDEO_FILES[@]} ]; then
    echo "Error: Array index ${SLURM_ARRAY_TASK_ID} is out of bounds"
    exit 1
fi

# Get the video file for this array task
INPUT_VIDEO_HOST_PATH="${VIDEO_FILES[${SLURM_ARRAY_TASK_ID}]}"
INPUT_VIDEO_FILENAME=$(basename "${INPUT_VIDEO_HOST_PATH}")

# Derive output filenames
VIDEO_BASENAME=$(basename "${INPUT_VIDEO_FILENAME}" .mp4)
OUTPUT_VIDEO_NAME="${VIDEO_BASENAME}_tracked.mp4"
OUTPUT_JSON_NAME="${VIDEO_BASENAME}_tracked.json"

# --- Path Derivations for Apptainer ---
# Container bind targets
PROJECT_CONTAINER_DIR="/opt/facetracker"
INPUT_VIDEO_CONTAINER_BIND_DIR="/input_videos"
OUTPUT_CONTAINER_BIND_DIR="/output"
MODEL_CONTAINER_BIND_DIR="/model_assets"

# Derive source directories and filenames for binds
INPUT_VIDEO_HOST_BASEDIR=$(dirname "${INPUT_VIDEO_HOST_PATH}")
MODEL_HOST_BASEDIR=$(dirname "${FACE_MODEL_HOST_PATH}")

# Path to the python script inside the container
PYTHON_SCRIPT_IN_CONTAINER="${PROJECT_CONTAINER_DIR}/scripts/run_pipeline.py"

# Paths for arguments to the python script (inside container)
VIDEO_ARG_PATH_IN_CONTAINER="${INPUT_VIDEO_CONTAINER_BIND_DIR}/${INPUT_VIDEO_FILENAME}"
OUTPUT_DIR_IN_CONTAINER="${OUTPUT_CONTAINER_BIND_DIR}"
FACE_MODEL_ARG_PATH_IN_CONTAINER="${MODEL_CONTAINER_BIND_DIR}/$(basename ${FACE_MODEL_HOST_PATH})"
POSE_MODEL_ARG_PATH_IN_CONTAINER="${MODEL_CONTAINER_BIND_DIR}/$(basename ${POSE_MODEL_HOST_PATH})"

# Full paths for output on the host
FULL_OUTPUT_VIDEO_PATH_ON_HOST="${OUTPUT_HOST_DIR}/${OUTPUT_VIDEO_NAME}"
FULL_OUTPUT_JSON_PATH_ON_HOST="${OUTPUT_HOST_DIR}/${OUTPUT_JSON_NAME}"
# --- End Path Derivations ---

# Create output directories if they don't exist
mkdir -p ../slurm_logs
mkdir -p "${OUTPUT_HOST_DIR}"

# Echo configuration
echo "Starting pipeline script for array task ${SLURM_ARRAY_TASK_ID}..."
echo "Processing video: ${INPUT_VIDEO_FILENAME}"
echo "SIF File: ${SIF_PATH}"
echo "Project Host Directory: ${PROJECT_HOST_DIR}"
echo "Input Video (Host): ${INPUT_VIDEO_HOST_PATH}"
echo "Face Model (Host): ${FACE_MODEL_HOST_PATH}"
echo "Pose Model (Host): ${POSE_MODEL_HOST_PATH}"
echo "Output Video (Host): ${FULL_OUTPUT_VIDEO_PATH_ON_HOST}"
echo "Output JSON (Host): ${FULL_OUTPUT_JSON_PATH_ON_HOST}"
echo "---"
echo "Python Script (Container): ${PYTHON_SCRIPT_IN_CONTAINER}"
echo "Video Path Arg (Container): ${VIDEO_ARG_PATH_IN_CONTAINER}"
echo "Output Dir Arg (Container): ${OUTPUT_DIR_IN_CONTAINER}"
echo "Face Model Path Arg (Container): ${FACE_MODEL_ARG_PATH_IN_CONTAINER}"
echo "Pose Model Path Arg (Container): ${POSE_MODEL_ARG_PATH_IN_CONTAINER}"
echo "---"

# Load Apptainer module (specific to HPC environment)
module load apptainer/1.1.9

# Run the Python script using Apptainer
apptainer exec --nv \
    --bind "${PROJECT_HOST_DIR}:${PROJECT_CONTAINER_DIR}" \
    --bind "${INPUT_VIDEO_HOST_BASEDIR}:${INPUT_VIDEO_CONTAINER_BIND_DIR}:ro" \
    --bind "${OUTPUT_HOST_DIR}:${OUTPUT_CONTAINER_BIND_DIR}" \
    --bind "${MODEL_HOST_BASEDIR}:${MODEL_CONTAINER_BIND_DIR}:ro" \
    "${SIF_PATH}" \
    python3 "${PYTHON_SCRIPT_IN_CONTAINER}" \
        --video_path "${VIDEO_ARG_PATH_IN_CONTAINER}" \
        --output_base_dir "${OUTPUT_DIR_IN_CONTAINER}" \
        --face_landmarker_model_path "${FACE_MODEL_ARG_PATH_IN_CONTAINER}" \
        --pose_landmarker_model_path "${POSE_MODEL_ARG_PATH_IN_CONTAINER}"

echo "Script finished for array task ${SLURM_ARRAY_TASK_ID}." 