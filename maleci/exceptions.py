class NoSelectionException(Exception):
    pass

class CancelException(Exception):
    pass

class NotFileException(Exception):
    pass

class NotFolderException(Exception):
    pass

class WrongVersionException(Exception):
    def __init__(self, version, options, name="version") -> None:
        super().__init__()

        self.version = version
        self.options = options
        self.name = name
    
    def __repr__(self) -> str:
        return f"WrongVersionException({self.name}={self.version}, options={self.options})"

    def __str__(self) -> str:
        return self.__repr__()

class NoArgumentException(Exception):
    pass

class VerificationCancelledException(Exception):
    pass

class NoError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    @staticmethod
    def really_not_an_exception(a, b):
        return a + str(ValueError(b))