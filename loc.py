#!/usr/bin/env python3

import os

blacklist = {'__pycache__', '.git', 'venv'}
total = 0
d = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in blacklist]
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loc = len(f.readlines())
                total += loc
                d.append((loc, file_path))
        except Exception as e:
            print(f"Could not read {file_path}: {e}")

for file in sorted(d, reverse=True):
    print(f"{file[1][2:]:30}  {file[0]}")
    
print(f"{'-' * 36}\n{'total':30}  {total}")
