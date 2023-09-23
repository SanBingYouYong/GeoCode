import cv2
import json
from typing import List, Tuple


class UIExternal():
    '''
    Read annotations image
        Binarize to (0, 255)
    Read component type selection information
        Load corresponding model
    Inference and save output params
    '''
    def __init__(self, default_image_info_path: str="./data/addon_out/annotations_info.json", 
                 inference_models_path: str="./data/outputs/models/",  # model .pt files to load  # TODO: find out .pt names... 
                 output_params_path: str="./data/outputs/annotations_param.csv") -> None:
        with open(default_image_info_path, mode='r', encoding='utf-8') as f:
            json_data = json.load(f)
        self.image_path = json_data["img_path"]
        self.type_info = json_data["type_info"]  # num in str
        self.inference_models_path = inference_models_path
        self.output_params_path = output_params_path

    def binarize_image_opencv(self, image_path: str=None, output_path: str=None, 
                              threshold=128, invert=True) -> None:
        '''
        Read, binarize and overwrite the image. 
        '''
        image_path = image_path if image_path else self.image_path
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        ret, binary_img = cv2.threshold(img, threshold, 255, threshold_type)
        output_path = output_path if output_path else self.image_path  # default overwrite
        cv2.imwrite(output_path, binary_img)


# ui = UIExternal()
# ui.binarize_image_opencv()
