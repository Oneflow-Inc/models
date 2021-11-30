import os
import sys


class Logger():
    def __init__(self, filename='log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()
