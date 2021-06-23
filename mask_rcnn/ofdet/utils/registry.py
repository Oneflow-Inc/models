import inspect


class Registry:
    """
    The Registry that supports name to module mapping.
    """

    def __init__(self, name):
        """
        Args:
            name (str): Registry name.
        """
        self._name = name
        self._module_map = {}

    def __contains__(self, name):
        return name in self._module_map

    @property
    def name(self):
        return self._name

    def get(self, name):
        module = self._module_map.get(name)
        if module is None:
            raise KeyError(
                "Module '{}' not found in '{}' registry.".format(name, self._name)
            )
        return module

    def _register_generic(self, name, module):
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError("module must be a class or a func.")
        if name in self._module_map:
            raise KeyError("'{}' is already registered in '{}'.".format(name, self._name))
        self._module_map[name] = module

    def register(self, module=None):
        if module is not None:
            name = module.__name__
            self._register_generic(name, module)
            return module

        # decorator
        def _register(module):
            name = module.__name__
            self._register_generic(name, module)
            return module

        return _register
