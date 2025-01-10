from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import InferencingSerializer
from .implementation import model_builder_class


# Create your views here.

class InferenceView(APIView):
    def post(self, request):
        serializer = InferencingSerializer(data=request.data)
        if serializer.is_valid():
            model_file_paths = [request.data.get("model_weight").temporary_file_path()]
            input_file_paths = [request.data.get("original_image").temporary_file_path()]

            contour_size = int(request.data.get("contour_size", 3))
            rgb = request.data.get("rgb", "255,0,0")
            is_grayscale = request.data.get("is_grayscale", False)

            # Build model
            model_builder = model_builder_class()
            model = model_builder.build(model_file_paths)

            # Perform inference
            result = model.infer(input_file_paths, contour_size=contour_size, rgb=rgb, is_grayscale=is_grayscale)

            return Response(result, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
