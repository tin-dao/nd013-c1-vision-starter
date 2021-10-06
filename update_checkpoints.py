import os
import re
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update checkpoint file for evaluation process')
    parser.add_argument('--source', required=True,
                        help='original checkpoint file')
    parser.add_argument('--dest', required=True,
                        help='new checkpoint file')
    parser.add_argument('--start_index', required=False, default=1, type=int,
                        help='Index of first checkpoint')
    parser.add_argument('--sleep', required=False, default=500, type=int,
                        help='Sleep time to wait for evaluation process')
    args = parser.parse_args()
    source = args.source
    dest = args.dest
    start_index = args.start_index
    sleep = args.sleep
    line = None
    timestamps = []

    with open(source, 'r') as f:
        line = f.read().splitlines()
    if line:
        for l in line:
            timestamp = re.search('all_model_checkpoint_timestamps:(.*)', l)
            if timestamp:
                timestamps.append(timestamp.group(1))
    
    for i in range(len(timestamps)):
        print(f'Update checkpoint with index {i + 1}...')
        with open(dest, 'w') as f:
            f.write(f'model_checkpoint_path: "ckpt-{i+1}"\n')
            for j in range(i + 1):
                f.write(f'all_model_checkpoint_paths: "ckpt-{start_index + j}"\n')
                f.write(f'all_model_checkpoint_timestamps:{timestamps[j]}\n')
        print(f'Wait evaluation process for {sleep}s...')
        time.sleep(sleep)
