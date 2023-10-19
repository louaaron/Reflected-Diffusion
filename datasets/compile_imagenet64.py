import os
import json
import PIL.Image
import sys
from pathlib import Path

PIL.Image.init()
source_dir = sys.argv[1]
output_dir = "imagenet-64x64"

def file_ext(name):
    return str(name).split('.')[-1]

def is_image_ext(fname):
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

print(input_images[0:10])
toplevel_names = [os.path.relpath(fname, source_dir).split('/')[0] for fname in input_images]
print(toplevel_names[0:10])
toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names)))}

def get_name(idx):
    idx_str = f'{idx:08d}'
    return f'{idx_str[:5]}/img{idx_str}.png'
labels = [[get_name(i), toplevel_indices[toplevel_name]] for i, toplevel_name in enumerate(toplevel_names)]

print(len(labels))
print(labels[0:10])

metadata = {'labels': labels}
data = json.dumps(metadata)

with open(os.path.join(output_dir, "dataset.json"), 'wb') as fout:
    if isinstance(data, str):
        data = data.encode('utf8')
    fout.write(data)
