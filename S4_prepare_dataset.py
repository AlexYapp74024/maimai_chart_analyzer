#%%
import os
from pathlib import Path
import random
import shutil
import yaml

random.seed(42)

(RAW_DATA_DIR := Path("data")).mkdir(exist_ok=True)
(DATASET_DIR := Path("dataset")).mkdir(exist_ok=True)

#%%
labels = [Path(l) for l in os.listdir(RAW_DATA_DIR / "labels")]
labels = random.sample(labels, len(labels))

size  = len(labels)
train = labels[:int(0.7 * size)]
valid = labels[len(train): int(0.8*size)]
test  = labels[len(valid) + len(train):]

assert train + valid + test == labels

#%%
def copy_label_and_image(label:str, dir:Path):
    label_path = Path(RAW_DATA_DIR / "labels" / label)
    image_path = Path(RAW_DATA_DIR / "images" / label).with_suffix(".webp")
    
    assert label_path.exists()
    assert image_path.exists()

    target_label_path = Path(DATASET_DIR / dir / "labels" / label_path.stem).with_suffix(label_path.suffix)
    target_image_path = Path(DATASET_DIR / dir / "images" / image_path.stem).with_suffix(image_path.suffix)

    target_label_path.parent.mkdir(exist_ok=True, parents=True)
    target_image_path.parent.mkdir(exist_ok=True, parents=True)

    shutil.copy(label_path, target_label_path)
    shutil.copy(image_path, target_image_path)

[copy_label_and_image(f, "train") for f in train]
[copy_label_and_image(f, "valid") for f in valid]
[copy_label_and_image(f, "test")  for f in test]
pass

#%%
with open(RAW_DATA_DIR / "data.yaml") as file:
    yaml_data = yaml.safe_load(file)

yaml_data["train"]= "train/images"
yaml_data["val"] =  "valid/images"
yaml_data["test"] = "test/images"

with open(DATASET_DIR / "data.yaml", "w") as file:
    yaml.safe_dump(yaml_data, file)

# %%
print(f"cd {Path('.').absolute()}")

# %%
from pathlib import Path
MODEL_FILE = list(Path("runs").glob("**/last.pt"))[-1]
DATASET_DIR = Path("dataset")

params = f"batch=12 epochs=70 imgsz=640 dropout=0.3 mixup=0.5 copy_paste=0.5"
print(f"yolo train model=yolov9c.pt data={(DATASET_DIR/'data.yaml').absolute()} {params}")
print(f"yolo train model={MODEL_FILE} data={(DATASET_DIR/'data.yaml').absolute()} {params} resume=True")
# %%
# %%
