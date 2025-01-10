from typing import TypedDict

class InferenceResult(TypedDict):
    data: str
    info: dict
    type: str

inference_metadata = {
    "inputs": ["image_file_path"],
    "outputs": ["InferenceResult"],
    "description": "Model for building segmentation and contour detection",
    "parameters": {
        "contour_size": "int, default=3",
        "rgb": "str, RGB color for contour visualization, e.g., '255,0,0'",
        "is_grayscale": "bool, if True visualizes in grayscale"
    }
}
