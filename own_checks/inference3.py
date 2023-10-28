import os
import sys
import json
import torch
import shutil
import traceback
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from functools import partial
import pytorch_lightning as pl
from subprocess import Popen, PIPE
from data.dataset_pc import DatasetPC
from data.dataset_sketch import DatasetSketch
from geocode.barplot_util import gen_and_save_barplot
from common.param_descriptors import ParamDescriptors
from geocode.geocode_util import InputType, get_inputs_to_eval, calc_prediction_vector_size
from geocode.geocode_model import Model
from torch.utils.data import DataLoader
# from chamfer_distance import ChamferDistance as chamfer_dist
from common.sampling_util import sample_surface
from common.file_util import load_obj, get_recipe_yml_obj
from common.point_cloud_util import normalize_point_cloud

sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/")


SINGLE_IMAGE_DATASET_DIR = "./datasets/ChairLess"
PHASE = "test"
EXP_NAME = "exp_geocode_chair"
MODELS_DIR = "./models"
BLENDER_DIR = os.path.expandvars("$BLENDER32")
BLEND_FILE = "./blends/procedural_chair.blend"


def save_as_obj_proc(pred_yml_file_path: Path, recipe_file_path: Path, results_dir: Path, out_dir: str, blender_exe: str, blend_file: str):
    target_obj_file_path = results_dir.joinpath(out_dir, f'{pred_yml_file_path.stem}.obj')
    print(f"Converting [{pred_yml_file_path}] to obj file [{target_obj_file_path}]")
    save_obj_script_path = Path(__file__).parent.joinpath('..', 'common', 'save_obj.py').resolve()
    cmd = [f'{str(Path(blender_exe).expanduser())}', f'{str(Path(blend_file).expanduser())}', '-b', '--python',
           f"{str(save_obj_script_path)}", '--',
           '--recipe-file-path', str(recipe_file_path),
           '--yml-file-path', str(pred_yml_file_path),
           '--target-obj-file-path', str(target_obj_file_path),
           '--ignore-sanity-check']
    print(" ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    process.wait()


recipe_file_path = Path(SINGLE_IMAGE_DATASET_DIR, 'recipe.yml')
if not recipe_file_path.is_file():
    raise Exception(f'No \'recipe.yml\' file found in path [{recipe_file_path}]')
recipe_yml_obj = get_recipe_yml_obj(str(recipe_file_path))

inputs_to_eval = get_inputs_to_eval(recipe_yml_obj)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
camera_angles_to_process = recipe_yml_obj['camera_angles_train'] + recipe_yml_obj['camera_angles_test']
camera_angles_to_process = [f'{a}_{b}' for a, b in camera_angles_to_process]

param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval, use_regression=False)
param_descriptors_map = param_descriptors.get_param_descriptors_map()
detailed_vec_size = calc_prediction_vector_size(param_descriptors_map)
print(f"Prediction vector length is set to [{sum(detailed_vec_size)}]")

# setup required dirs
required_dirs = ['barplot',
                    'yml_gt', 'yml_predictions_pc', 'yml_predictions_sketch',
                    'obj_gt', 'obj_predictions_pc', 'obj_predictions_sketch',
                    'render_gt', 'render_predictions_pc', 'render_predictions_sketch',
                    'sketch_gt']
test_dir = Path(SINGLE_IMAGE_DATASET_DIR, PHASE)
test_dir_obj_gt = test_dir.joinpath('obj_gt')
results_dir = test_dir.joinpath(f'results_{EXP_NAME}')
results_dir.mkdir(exist_ok=True)
for dir in required_dirs:
    results_dir.joinpath(dir).mkdir(exist_ok=True)

# save the recipe to the results directory
shutil.copy(recipe_file_path, results_dir.joinpath('recipe.yml'))

# find the best checkpoint (the one with the highest epoch number out of the saved checkpoints)
exp_dir = Path(MODELS_DIR, EXP_NAME)
best_model_and_highest_epoch = None
highest_epoch = 0
for ckpt_file in exp_dir.glob("*.ckpt"):
    file_name = ckpt_file.name
    if 'epoch' not in file_name:
        continue
    epoch_start_idx = file_name.find('epoch') + len('epoch')
    epoch = int(file_name[epoch_start_idx:epoch_start_idx + 3])
    if epoch > highest_epoch:
        best_model_and_highest_epoch = ckpt_file
        highest_epoch = epoch
print(f'Best model with highest epoch is [{best_model_and_highest_epoch}]')

batch_size = 1
test_dataloaders = []
test_dataloaders_types = []
test_dataset_sketch = DatasetSketch(inputs_to_eval, param_descriptors_map,
                                    camera_angles_to_process, False,
                                    SINGLE_IMAGE_DATASET_DIR, PHASE)
test_dataloader_sketch = DataLoader(test_dataset_sketch, batch_size=batch_size, shuffle=False,
                                    num_workers=2, prefetch_factor=2)
test_dataloaders.append(test_dataloader_sketch)
test_dataloaders_types.append('sketch')

pl_model = Model.load_from_checkpoint(str(best_model_and_highest_epoch), batch_size=1,
                                        param_descriptors=param_descriptors, results_dir=results_dir,
                                        test_dir=test_dir, models_dir=MODELS_DIR,
                                        test_dataloaders_types=test_dataloaders_types, test_input_type=[InputType.sketch],
                                        exp_name=EXP_NAME)

trainer = pl.Trainer(gpus=1)
trainer.test(model=pl_model, dataloaders=test_dataloaders, ckpt_path=best_model_and_highest_epoch)

# report average inference time
avg_inference_time = pl_model.inference_time / pl_model.num_inferred_samples
print(f"Average inference time for [{pl_model.num_inferred_samples}] samples is [{avg_inference_time:.3f}]")

# save the validation and test bar-plots as image
barplot_target_dir = results_dir.joinpath('barplot')
for barplot_type in ['val', 'test']:
    barplot_json_path = Path(MODELS_DIR, EXP_NAME, f'{barplot_type}_barplot_top_1.json')
    if not barplot_json_path.is_file():
        print(f"Could not find barplot [{barplot_json_path}] skipping copy")
        continue
    barplot_target_image_path = barplot_target_dir.joinpath(f'{barplot_type}_barplot.png')
    title = "Validation Accuracy" if barplot_type == 'val' else "Test Accuracy"
    gen_and_save_barplot(barplot_json_path, title, barplot_target_image_path=barplot_target_image_path)
    shutil.copy(barplot_json_path, barplot_target_dir.joinpath(barplot_json_path.name))
