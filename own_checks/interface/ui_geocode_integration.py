import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context

import sys
sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/own_checks/interface")
from ui_external import resize_and_convert

'''
Template for getting started with a Blender interface quickly. 
Bind methods to those `execute` methods. 
Remember to replace Blender's default python directory. 
    e.g. remove built-in python folder, start blender with conda env activated (python path overwrite). 
        or modify default python path manually. 
'''


class CaptureAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.capture_annotation_operator"
    bl_label = "Capture Annotation"

    def execute(self, context: Context):
        print("You've called Capture Annotation.")
        scene = context.scene
        img_path = bpy.path.abspath(scene.annotation_image_path)
        if os.path.isdir(img_path):
            img_path = os.path.join(img_path, "annotation_image.png")
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.opengl(write_still=True)
        resize_and_convert(img_path)
        return {"FINISHED"}


class ClearAllAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.clear_all_annotation_operator"
    bl_label = "Clear All Annotation"
    
    def execute(self, context: Context):
        print("You've called Clear All Annotation.")
        scene = context.scene
        for annotation_layer in scene.grease_pencil.layers:
            annotation_layer.clear()
        return {'FINISHED'}


class GeoCodeInterfacePanel(bpy.types.Panel):
    bl_label = "GeoCode Interface Panel"
    bl_idname = "SIUI_PT_GeoCodeInterfacePanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context: Context):
        layout = self.layout
        scene = context.scene
        # add elements to ui
        layout.prop(scene, "annotation_image_path")
        layout.operator("object.capture_annotation_operator")
        layout.operator("object.clear_all_annotation_operator")

def register():
    bpy.types.Scene.annotation_image_path = bpy.props.StringProperty(
        name="Annotation Image Path Property", 
        subtype='FILE_PATH',
        default="//annotation_image"
    )
    
    bpy.utils.register_class(CaptureAnnotationOperator)
    bpy.utils.register_class(ClearAllAnnotationOperator)
    bpy.utils.register_class(GeoCodeInterfacePanel)

def unregister():
    del bpy.types.Scene.annotation_image_path
    
    bpy.utils.unregister_class(CaptureAnnotationOperator)
    bpy.utils.unregister_class(ClearAllAnnotationOperator)
    bpy.utils.unregister_class(GeoCodeInterfacePanel)

if __name__ == "__main__":
    register()
