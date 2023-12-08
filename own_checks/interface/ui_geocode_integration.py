import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context

# from own_checks.inference.inference3 import gc_single_image_inference

# from pathlib import Path
# def import_parents(level=1):
#     global __package__
#     file = Path(__file__).resolve()
#     parent, top = file.parent, file.parents[level]
#     sys.path.append(str(top))
#     try:
#         sys.path.remove(str(parent))
#     except ValueError:
#         pass
#     __package__ = '.'.join(parent.parts[len(top.parts):])
#     importlib.import_module(__package__)
# import_parents(level=1)

from PIL import Image

sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/")

from common.bpy_util import select_shape, get_geometric_nodes_modifier
from common.input_param_map import get_input_param_map, load_shape_from_yml
from common.file_util import get_recipe_yml_obj

import torch
import shutil
from pathlib import Path
import pytorch_lightning as pl
from subprocess import Popen, PIPE
from data.dataset_sketch import DatasetSketch
from geocode.barplot_util import gen_and_save_barplot
from common.param_descriptors import ParamDescriptors
from geocode.geocode_util import (
    InputType,
    get_inputs_to_eval,
    calc_prediction_vector_size,
)
from geocode.geocode_model import Model
from torch.utils.data import DataLoader
from common.file_util import get_recipe_yml_obj

import logging

from own_checks.inference_methods import GCDomain, gc_single_image_inference_entrypoint
from own_checks.param2obj_methods import param2obj_entrypoint



def checkDataset(dataloader: DataLoader):
    # iterate through dataloader and log data
    logging.info(f"len(dataloader): {len(dataloader)}")
    for batch_idx, batch in enumerate(dataloader):
        logging.info(f"batch_idx: {batch_idx}")
        logging.info(f"batch: {batch}")


def resize_and_convert(img_path: str) -> None:
    # Open an image
    image = Image.open(img_path)
    # Resize the image (e.g., to 300x300 pixels)
    resized_image = image.resize((224, 224))
    # convert to grayscale and save as RGB
    binarized_image = resized_image.convert("L")
    # invert the image
    binarized_image = Image.eval(binarized_image, lambda x: 255 - x)
    # convert back to RGB
    binarized_image = binarized_image.convert("RGB")
    binarized_image.save(img_path)
    print(f"PIL saved img to {img_path}")


class CaptureAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.capture_annotation_operator"
    bl_label = "Capture Annotation & GeoCode"

    def execute(self, context: Context):
        print("You've called Capture Annotation & GeoCode Inference.")

        # hide "procedural shape"
        bpy.data.objects["procedural shape"].hide_viewport = True

        scene: bpy.types.Scene = context.scene
        img_path = "./datasets/SingleImg/test/sketches/single_img_-30.0_15.0.png"
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.opengl(write_still=True)
        resize_and_convert(img_path)

        gc_single_image_inference_entrypoint(GCDomain.CHAIR)
        param2obj_entrypoint(GCDomain.CHAIR)

        # bring it back in
        bpy.data.objects["procedural shape"].hide_viewport = False
        return {"FINISHED"}


class ClearAllAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.clear_all_annotation_operator"
    bl_label = "Clear All Annotation"

    def execute(self, context: Context):
        print("You've called Clear All Annotation.")
        scene = context.scene
        for annotation_layer in scene.grease_pencil.layers:
            annotation_layer.clear()
        return {"FINISHED"}


class GeoCodeInterfacePanel(bpy.types.Panel):
    bl_label = "GeoCode Interface Panel"
    bl_idname = "SIUI_PT_GeoCodeInterfacePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"

    def draw(self, context: Context):
        layout = self.layout
        scene = context.scene
        # add elements to ui
        # layout.prop(scene, "annotation_image_path")
        layout.prop(scene, "slider_value", text="View Angle")
        layout.operator("object.capture_annotation_operator")
        layout.operator("object.clear_all_annotation_operator")


def update_slider_value(self, context):
    print("slider value updated")
    scene = context.scene
    print(scene.slider_value)
    # update z rotation of "CameraTrack" object accordingly
    bpy.data.objects["CameraTrack"].rotation_euler[2] = (
        scene.slider_value * 3.1415926 / 180.0
    )
    print("updated camera rotation")


def register():
    # bpy.types.Scene.annotation_image_path: bpy.types.StringProperty = bpy.props.StringProperty(
    #     name="Annotation Image Path Property",
    #     subtype='FILE_PATH',
    #     default="//datasets//SingleImg//test//sketches//single_img_-30.0_15.0.png"
    # )

    bpy.types.Scene.slider_value = bpy.props.FloatProperty(
        name="View Angle", default=0.0, min=0.0, max=90.0, update=update_slider_value
    )

    bpy.utils.register_class(CaptureAnnotationOperator)
    bpy.utils.register_class(ClearAllAnnotationOperator)
    bpy.utils.register_class(GeoCodeInterfacePanel)


def unregister():
    # del bpy.types.Scene.annotation_image_path

    bpy.utils.unregister_class(CaptureAnnotationOperator)
    bpy.utils.unregister_class(ClearAllAnnotationOperator)
    bpy.utils.unregister_class(GeoCodeInterfacePanel)


if __name__ == "__main__":
    register()
