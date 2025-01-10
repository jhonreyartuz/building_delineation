from matplotlib import pyplot as plt
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
from .inference_metadata import InferenceResult

class Model:
    def __init__(self, model_weights_path, encoder="efficientnet-b4", activation="sigmoid"):
        self.encoder = encoder
        self.activation = activation
        try:
            self.model = self.load_model(model_weights_path)
            print("[INFO] Model initialized successfully with encoder: ", self.encoder)
        except Exception as e:
            print("[ERROR] Failed to initialize the model: ", str(e))
            raise

    def load_model(self, model_weights_path):
        try:
            print("[INFO] Loading model weights from: ", model_weights_path)
            model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights="imagenet",
                classes=2,
                decoder_attention_type='scse',
                activation=self.activation
            )
            best_model = torch.load(model_weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(best_model['state_dict'])
            model.eval()
            print("[INFO] Model wieghts loaded successfully.")
            return model
        except FileNotFoundError:
            print("[ERROR] Model wieghts file not found.")
            raise
        except Exception as e:
            print("[Error] An error occurred while loading the model weights", str(e))

    def preprocess_image(self, image_path):
        try:
            print("[INFO] Processing image: ", image_path)
            transform = transforms.Compose([
                transforms.Resize((736, 1280)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = Image.open(image_path).convert('RGB')
            print("[INFO] Image preprocessed successfully.")
            return transform(image).unsqueeze(0)
        except FileNotFoundError:
            print("[ERROR] Image file not found:", image_path)
            raise
        except Exception as e:
            print("[ERROR] An error occurred during image preprocessing:", str(e))
            raise



    def predict(self, image_tensor):
        try:
            print("[INFO] Running prediction on the image tensor.")
            with torch.no_grad():
                prediction = self.model(image_tensor)
            print("[INFO] Prediction completed successfully.")
            return prediction
        except Exception as e:
            print("[ERROR] An error occurred during prediction:", str(e))
            raise


    def count_contours(self, image_path, prediction):
        try:
            print("[INFO] Counting contours in the prediction mask.")
            original_image = cv2.imread(image_path)
            prediction_array = prediction.cpu().numpy()[0, 1]
            resized_prediction = cv2.resize(prediction_array, (original_image.shape[1], original_image.shape[0]))
            prediction_mask = (resized_prediction > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(prediction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print("[INFO] Contours counted successfully. Total contours: ",len(contours))
            return len(contours)
        except Exception as e:
            print("[ERROR] An error occurred while counting contours:", str(e))
            raise
        

    def infer(self, input_file_paths, contour_size=3, rgb="255,0,0", is_grayscale=False):
        try:
            print("[INFO] Starting inference.")
            original_image_path = input_file_paths[0]
            image_tensor = self.preprocess_image(original_image_path)
            prediction = self.predict(image_tensor)
            contours_count = self.count_contours(original_image_path, prediction)

            result_image_path = self.visualize_prediction(original_image_path, prediction, contour_size, rgb, is_grayscale)
            print("[INFO] Inference completed successfully. Result saved at:", result_image_path)
            return InferenceResult(
                data=result_image_path,
                info={"contours": contours_count},
                type="image/png"
            )
        except Exception as e:
            print("[ERROR] An error occurred during inference:", str(e))
            raise

    def visualize_prediction(self, image_path, prediction, contour_size, rgb, is_grayscale):
        try:
            print("[INFO] Visualizing prediction.")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join("media/result_bd", timestamp)
            os.makedirs(output_dir, exist_ok=True)
            result_image_path = os.path.join(output_dir, "result.png")

            if not is_grayscale:
                original_image = cv2.imread(image_path)
                prediction_array = prediction.cpu().numpy()[0, 1]
                resized_prediction = cv2.resize(prediction_array, (original_image.shape[1], original_image.shape[0]))
                prediction_mask = (resized_prediction > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(prediction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rgb_tuple = tuple(map(int, rgb.split(',')))
                for contour in contours:
                    cv2.drawContours(original_image, [contour], -1, rgb_tuple[::-1], contour_size)
                cv2.imwrite(result_image_path, original_image)
            else:
                plt.imshow(prediction.cpu().numpy()[0, 1], cmap="gray")
                plt.axis('off')
                plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            print("[INFO] Visualization completed successfully. Output saved at:", result_image_path)
            return result_image_path
        
        except Exception as e:
            print("[ERROR] An error occurred during visualization:", str(e))
            raise
