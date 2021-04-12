from pathlib import Path
from utils.serializer import load_dict, save_dict


class ConfigError(Exception):
    pass


class BaseConfig:
    _params = {}
    _ERROR_MSG = "'{cls}' object has no attribute '{attr_name}'"

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_file(cls, path):
        config_dict = load_dict(path)
        return cls(**config_dict)

    def __init__(self, **kwargs):
        self.set_defaults()
        self.update(**kwargs)

    def __getattr__(self, name):
        if name not in self:
            cls_name = self.__class__.__name__
            raise AttributeError(
                self._ERROR_MSG.format(cls=cls_name, attr_name=name))
        return self._params[name]

    def __setattr__(self, name, value):
        if name not in self:
            cls_name = self.__class__.__name__
            raise AttributeError(
                self._ERROR_MSG.format(cls=cls_name, attr_name=name))
        self._params[name] = value

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __contains__(self, name):
        return name in self._params

    def update(self, **kwargs):
        for opt_name, opt_value in kwargs.items():
            self.__setattr__(opt_name, opt_value)

    def set_defaults(self):
        raise NotImplementedError


class BaseConfigWithSerializer(BaseConfig):
    serializer = 'yaml'

    @classmethod
    def from_rundir(cls, path):
        path = path / f"config.{cls.serializer}"
        return cls.from_file(path)

    def __init__(self, **kwargs):
        if self.serializer not in ['json', 'yaml', 'pickle']:
            raise ConfigError(f"Unknown serializer '{self.serializer}'")
        super().__init__(**kwargs)

    def save(self, path):
        save_dict(self._params, path, self.serializer)
