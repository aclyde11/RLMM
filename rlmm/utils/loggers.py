def make_message_writer(verbose, class_name):
    class MessageWriter(object):
        def __init__(self, method_name):
            self.class_name = class_name
            self.verbose = verbose
            self.method_name = method_name

        def log(self, *args, **kwargs):
            if self.verbose:
                print("[{}:{}]".format(self.class_name, self.method_name), *args, **kwargs)

        def __enter__(self):
            self.log("Entering")
            return self

        def __exit__(self, *args, **kwargs):
            self.log("Exiting")

    return MessageWriter