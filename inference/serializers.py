from rest_framework import serializers
from .models import Inference

class InferencingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Inference
        fields = ["original_image", "model_weight", "is_grayscale", "contour_size", "rgb"]
