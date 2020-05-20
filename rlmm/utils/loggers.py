import sys

def make_message_writer(verbose_, class_name_):
    class MessageWriter(object):
        class_name = class_name_

        def __init__(self, method_name, verbose=None, enter_message=True):
            if verbose is None:
                self.verbose = verbose_
            else:
                self.verbose = verbose
            self.method_name = method_name
            self.enter_message = enter_message

        def log(self, *args, **kwargs):
            if self.verbose:
                print("INFO [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs)

        def error(self, *args, **kwargs):
            print("ERROR [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs, file=sys.stderr)

        def failure(self, *args, exit_all=False, **kwargs):
            print("FAILURE [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs, file=sys.stderr)
            if exit_all:
                exit()

        @classmethod
        def static_failure(cls, method_name, *args, exit_all=False, **kwargs):
            print("FAILURE [{}:{}]".format(cls.class_name, method_name), *args, **kwargs, file=sys.stderr)
            if exit_all:
                exit()

        def __enter__(self):
            if self.enter_message:
                self.log("Entering")
            return self

        def __exit__(self, *args, **kwargs):
            if self.enter_message:
                self.log("Exiting")

    return MessageWriter