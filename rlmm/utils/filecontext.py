import os

class FileContext:
    def __init__(self, simulation_workers=1, tmpdir=None):
        self.n = simulation_workers
        assert(self.n > 0 and self.n < 16)

        self.tempdir = tmpdir
        if self.tempdir[-1] == '/':
            self.tempdir = self.tempdir[:-1]
        assert(os.path.isdir(self.tempdir))
        if self.n == 1:
            self.tempdirs = [self.tempdir]
        else:
            assert(False) #not implemented yet
        self.mkdirs(self.tempdir)

        self.curstep = 0

    def mkdirs(self, dirs):
        for dir in dirs:
            try:
                os.makedirs(dir, exist_ok=False)
            except OSError as e:
                print(e.strerror)
                exit()

    def reset_steps(self):
        self.curstep = 0

    def start_step(self, step=None):
        if step is None:
            self.curstep = self.curstep + 1
        else:
            self.curstep = step
        for i in range(len(self.tempdirs)):
            self.tempdirs[i] = f"{self.tempdir}/{self.curstep}"
        self.mkdirs(self.tempdirs)

    def get_folder(self, id=0, main_context=False, step_context=-1):
        assert(id >= 0 and id < self.n)
        if main_context:
            return self.tempdir
        elif step_context < 0:
            return self.tempdirs[id]
        else:
            return f"{self.tempdir}/{step_context}"

    def __call__(self, *args, **kwargs):
        self.get_folder(*args, **kwargs)