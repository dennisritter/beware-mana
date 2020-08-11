"""An abstract loader class for specific sequence loaders."""


class Transformer:
    def __init__(self, transforms: list):
        self.transforms = transforms
