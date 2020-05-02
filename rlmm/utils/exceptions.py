class BadConfigError(Exception):
    def __init__(self, *args, **kwargs):
        default_msg = 'Bad configuration file'
        if args or kwargs:           # Allows for custom error messages
            super().__init__(*args, **kwargs)
        else:
            super().__init__(default_msg)