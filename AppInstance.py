from catalog.CacheModels import CacheModels
from catalog.CatalogueEntities import CatalogueEntities


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class AppInstance(metaclass=SingletonMeta):

    def __init__(self):
        self.__catalogueEntities = CatalogueEntities()
        self.__cacheModels = CacheModels()

    def getCatalogueEntities(self) -> CatalogueEntities:
        return self.__catalogueEntities

    def getCacheModels(self) -> CacheModels:
        return self.__cacheModels



