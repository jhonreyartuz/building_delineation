from django.db import models

# Create your models here.

class Inference(models.Model):
    original_image = models.ImageField(upload_to='original_bd/')
    result_image = models.ImageField(upload_to='result_bd/')
    model_weight = models.FileField(upload_to='weights/')
    is_grayscale = models.BooleanField(default=False)
    contour_size = models.IntegerField(default=3)
    rgb = models.CharField(max_length=11)

    def __str__(self):
        return f'Inference for image {self.original_image}'
