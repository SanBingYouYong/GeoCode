import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context

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

def resize_and_convert(img_path: str) -> None:
    # Open an image
    image = Image.open(img_path)
    # Resize the image (e.g., to 300x300 pixels)
    resized_image = image.resize((224, 224))
    # Convert the image to grayscale
    grayscale_image = resized_image.convert('L')
    # Save the grayscale image
    grayscale_image.save(img_path)
    print(f"PIL saved img to {img_path}")




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
        default="//annotation_image.png"
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
