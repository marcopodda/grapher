from pathlib import Path


class PathMixin:
    def __init__(self, root):
        self.__class__.root = root

    @classmethod
    def path(self, *parts):
        return Path(self.root, *parts)
