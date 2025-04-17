from os import mkdir
from os.path import isdir

REQ_DIRS = ["build", "sample_error", "src/train/data", "src/train/models"]

for dir in REQ_DIRS:
    if not isdir(dir):
        mkdir(dir)
