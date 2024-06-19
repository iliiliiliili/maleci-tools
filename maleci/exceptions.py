

class NoSelectionException(Exception):
    pass

class CancelException(Exception):
    pass

class NotFileException(Exception):
    pass

class NotFolderException(Exception):
    pass

class WrongVersionException(Exception):
    pass

class NoArgumentException(Exception):
    pass

class NoError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    @staticmethod
    def really_not_an_exception(a, b):
        return a + str(ValueError(b))