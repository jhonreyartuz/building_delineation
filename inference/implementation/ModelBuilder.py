from .Model import Model

class ModelBuilder:
    def build(self, model_file_paths):
        model_weights_path = model_file_paths[0]
        return Model(model_weights_path)
