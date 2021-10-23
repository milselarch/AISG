class blob(object):
    def __init__(self):
        self._name = 1

    @property
    def name(self):
        return self._name

    # property setter
    @name.setter
    def name(self, value, a):
        if isinstance(value, str) and value != "":
            self._name = value