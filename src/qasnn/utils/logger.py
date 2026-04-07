import csv

class MetricsLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.header_written = False
        self.columns = []

    def log(self, **kwargs):
        if not self.header_written:
            self.columns = list(kwargs.keys())
            self.writer.writerow(self.columns)
            self.header_written = True
        row = [kwargs.get(col, '') for col in self.columns]
        self.writer.writerow(row)

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()

def initialize_metrics_logger(filename):
    return MetricsLogger(filename)

def log_metrics(logger, **kwargs):
    logger.log(**kwargs)