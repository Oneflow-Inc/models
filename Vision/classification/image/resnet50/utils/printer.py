import pandas as pd
from collections import namedtuple


class Printer(object):
    def __init__(self, field_names, print_format="table", persistent_file=None):
        assert print_format in ("table", "normal")

        self.field_names_ = field_names
        self.format_ = print_format
        self.records_ = []
        self.handlers_ = dict()
        self.str_lens_ = dict()
        self.title_printed_ = False

        if persistent_file is not None:
            self.csv_ = open(persistent_file, "a")
        else:
            self.csv_ = None

        self.Record = None

    def __def__(self):
        if self.csv_ is not None:
            self.csv_.close()

    def finish(self):
        err = f"{len(self.field_names_)} vs. {len(self.handlers_)}"
        assert len(self.field_names_) == len(self.handlers_), err
        err = f"{len(self.field_names_)} vs. {len(self.str_lens_)}"
        assert len(self.field_names_) == len(self.str_lens_), err
        for fname in self.field_names_:
            assert fname in self.handlers_, f"{fname} handler not register"
            assert fname in self.str_lens_, f"{fname} str_len not register"

        self.Record = namedtuple("Record", self.field_names_)
        # DEBUG(zwx):
        # dummy = self.Record(*(["-"] * len(self.field_names_)))
        # df = pd.DataFrame(dummy)
        # if self.persistent_file_ is not None:
        #     df.to_csv(self.persistent_file_, mode='a', header=True)

    def record(self, *args, **kwargs):
        assert self.Record is not None
        r = self.Record(*args, **kwargs)
        self.records_.append(r)

    def register_handler(self, field, handler):
        assert callable(handler)
        self.handlers_[field] = handler

    def register_str_len(self, field, str_len):
        assert isinstance(str_len, int)
        self.str_lens_[field] = str_len

    def reset_records(self):
        self.records_ = []

    def print_table_title(self):
        fields = ""
        sep = ""

        for fname in self.field_names_:
            str_len = self.str_lens_[fname]
            fields += "| {} ".format(fname.ljust(str_len))
            sep += f"| {'-' * str_len} "

        fields += "|"
        sep += "|"
        print(fields)
        print(sep)
        self.title_printed_ = True

    def reset_title_printed(self):
        self.title_printed_ = False

    def print(self):
        df = pd.DataFrame(self.records_)
        fields = []
        for fname in self.field_names_:
            assert fname in self.handlers_
            handler = self.handlers_[fname]
            field_value = handler(df[fname])
            fields.append(field_value)

        if self.format_ == "table":
            if not self.title_printed_:
                self.print_table_title()

            record = ""
            for i, str_len in enumerate(self.str_lens_.values()):
                record += "| {} ".format(str(fields[i]).ljust(str_len))

            record += "|"
            print(record)

        elif self.format_ == "normal":
            record = ""

            for i, fval in enumerate(fields):
                fname = self.field_names_[i]
                record += f"{fname}: {fval}, "

            print(record)
        else:
            raise ValueError

        if self.csv_ is not None:
            df.to_csv(self.csv_, header=False)

        self.reset_records()
