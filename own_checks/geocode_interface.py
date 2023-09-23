import bpy
import os

class SaveAnnotationsOperator(bpy.types.Operator):
    bl_idname = "object.save_annotations_operator"
    bl_label = "Save Annotations"
    
    def execute(self, context):
        scene = bpy.context.scene
        img_path = bpy.path.abspath(scene.annotations_save_path)
        img_filename = scene.annotations_filename
        full_img_path = os.path.join(img_path, img_filename)
        
        option_info_path = bpy.path.abspath(scene.option_info_save_path)  # 获取选项信息保存路径
        option_info_filename = "option_info.txt"
        full_option_info_path = os.path.join(option_info_path, option_info_filename)
        
        # 保存图像
        bpy.context.scene.render.filepath = full_img_path
        bpy.ops.render.opengl(write_still=True)
        
        # 保存选项信息
        with open(full_option_info_path, "w") as f:
            f.write(f"Image Filename: {img_filename}\n")
            f.write(f"Selected Option Index: {scene.selected_option_index}\n")
        
        return {'FINISHED'}

class OptionPanel(bpy.types.Panel):
    bl_label = "Options Panel"
    bl_idname = "PT_OptionPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        
        layout.label(text="Select an option:")
        layout.prop(scene, "selected_option_index", expand=True)
        layout.prop(scene, "option_info_save_path")  # 添加选项信息保存路径选项

class SaveAnnotationsPanel(bpy.types.Panel):
    bl_label = "Save Annotations Panel"
    bl_idname = "PT_SaveAnnotationsPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        
        layout.prop(scene, "annotations_save_path")
        layout.prop(scene, "annotations_filename")
        layout.operator("object.save_annotations_operator")

def register():
    bpy.types.Scene.annotations_save_path = bpy.props.StringProperty(
        name="Save Path",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.annotations_filename = bpy.props.StringProperty(
        name="Filename",
        default="annotations_image.png"
    )
    
    bpy.types.Scene.option_info_save_path = bpy.props.StringProperty(  # 添加选项信息保存路径属性
        name="Option Info Save Path",
        subtype='DIR_PATH'
    )
    
    bpy.types.Scene.selected_option_index = bpy.props.EnumProperty(
        items=[("0", "Building Mass", "Building Mass"),
               ("1", "Roof", "Roof"),
               ("2", "Window", "Window"),
               ("3", "Ledge", "Ledge")],
        name="Selected Option",
        default="0"
    )
    
    bpy.utils.register_class(SaveAnnotationsOperator)
    bpy.utils.register_class(OptionPanel)
    bpy.utils.register_class(SaveAnnotationsPanel)

def unregister():
    del bpy.types.Scene.annotations_save_path
    del bpy.types.Scene.annotations_filename
    del bpy.types.Scene.option_info_save_path
    del bpy.types.Scene.selected_option_index
    
    bpy.utils.unregister_class(SaveAnnotationsOperator)
    bpy.utils.unregister_class(OptionPanel)
    bpy.utils.unregister_class(SaveAnnotationsPanel)

if __name__ == "__main__":
    register()
