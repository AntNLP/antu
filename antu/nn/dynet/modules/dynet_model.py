def dy_model(cls):

    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        return cls(model, *spec)

    cls.from_spec, cls.param_collection = from_spec, param_collection
    return cls
