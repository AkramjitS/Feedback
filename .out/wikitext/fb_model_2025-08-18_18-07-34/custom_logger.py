import os
import csv
import sys

class Logger:
    def __init__(self):
        self.log_file_path = None
        self.csv_file_path = None
        self.log_buffer = []
        self.csv_buffer = []
        self.csv_header = None

    def set_log_file(self, file_path):
        self.log_file_path = file_path
        if self.log_buffer:
            self._flush_log_buffer()

    def set_csv_file(self, file_path, header=None):
        self.csv_file_path = file_path
        self.csv_header = header
        if self.csv_buffer:
            self._flush_csv_buffer()

    def _flush_log_buffer(self):
        if self.log_file_path and self.log_buffer:
            with open(self.log_file_path, 'a') as f:
                for line in self.log_buffer:
                    f.write(line + '\n')
            self.log_buffer = []

    def _flush_csv_buffer(self):
        if self.csv_file_path and self.csv_buffer:
            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if self.csv_header and (not os.path.exists(self.csv_file_path) or os.stat(self.csv_file_path).st_size == 0):
                    writer.writerow(self.csv_header)
                    self.csv_header = None
                writer.writerows(self.csv_buffer)
            self.csv_buffer = []

    def log(self, message):
        print(message)
        if self.log_file_path:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
        else:
            self.log_buffer.append(message)

    def log_csv(self, row_data):
        # not printing row data as log should already be printing it in the appropriate output format
        #print(row_data)
        if self.csv_file_path:
            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if self.csv_header and (not os.path.exists(self.csv_file_path) or os.stat(self.csv_file_path).st_size == 0):
                    writer.writerow(self.csv_header)
                    self.csv_header = None
                writer.writerow(row_data)
        else:
            self.csv_buffer.append(row_data)