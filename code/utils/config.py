import os, sys
from huggingface_hub import login

## primary directories
BASE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/'
CACHE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/models/'
DATASETS_DIR = '/dartfs/rc/lab/F/FinnLab/datasets/'
SCRATCH_DIR = '/dartfs-hpc/scratch/f003rjw/'

## secondary directories
SUBMIT_DIR = os.path.join(BASE_DIR, 'code', 'submit_scripts')
LOGS_DIR = os.path.join(BASE_DIR, 'derivatives', 'logs')

# SET MODELS DIRECTORIES
os.environ['GENSIM_DATA_DIR'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['TORCH_HOME'] = CACHE_DIR

# Strings to append at start of run for dSQ
DSQ_MODULES	 = [
    'module load cuda/11.2',
    'module load openmpi',
    'module load workbench/1.50',
    'source /optnfs/common/miniconda3/etc/profile.d/conda.sh',
    'conda activate analysis' # analysis
]

DSQ_MODULES = ''.join([f'{module}; ' for module in DSQ_MODULES])