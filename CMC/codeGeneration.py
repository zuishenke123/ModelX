import collections
import os
import textwrap


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    def __new__(cls, fileName=None, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    return get_instance


@singleton
class commonGenerateHelper:
    def __init__(self, fileName=None, *args, **kwargs):
        if fileName:
            self.fileName = fileName
            self.ids = collections.defaultdict(bool)
            # Indicates whether the template code has been written
            self.code_written = False

    def generate_model_code(self, code, target_api, targetClass="paddle"):
        if not self.code_written:
            CODE_CONTENT = textwrap.dedent(
                """
                #the  Common model file template, please Don't edit it!
                import {}
                """.format(targetClass)
            )
            # Create the initial file
            if not os.path.exists(os.path.dirname(self.fileName)):
                os.makedirs(os.path.dirname(self.fileName))

            with open(self.fileName, "w") as file:
                file.write(CODE_CONTENT)

            self.code_written = True

        if self.ids[target_api] == 0:
            with open(self.fileName, "a") as file:
                file.write(code)

        self.ids[target_api] += 1

    def generate_pythonLibrary_code(self, ):
        pass

    def generate_CLibrary_code(self):
        pass
