import time


class CodeTimer:
    format_text = 'Code block {} took: {} ms'

    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        print(CodeTimer.format_text.format(self.name, str(self.took)))
