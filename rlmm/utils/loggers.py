import sys

def make_message_writer(verbose: bool, class_name_: str):
    class MessageWriter(object):
        class_name = class_name_

        def __init__(self, method_name: str) -> None:
            self.verbose = verbose
            self.method_name = method_name

        def log(self, *args, **kwargs):
            if self.verbose:
                print("INFO [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs)

        def error(self, *args, **kwargs):
            print("ERROR [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs, file=sys.stderr)

        def failure(self, *args, exit_all: Optional[bool] = False, **kwargs):
            print("FAILURE [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs, file=sys.stderr)
            if exit_all:
                exit()

        @classmethod
        def static_failure(cls, method_name: str, *args, exit_all: Optional[bool] = False, **kwargs):
            print("FAILURE [{}:{}]".format(cls.class_name, method_name), *args, **kwargs, file=sys.stderr)
            if exit_all:
                exit()

        def __enter__(self):
            self.log("Entering")
            return self

        def __exit__(self, *args, **kwargs):
            self.log("Exiting")

    return MessageWriter
