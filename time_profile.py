import time
import logging


class TimeProfile:
    class LabelData:
        def __init__(self, name: str, time_taken: float):
            self.name = name
            self.time_taken = time_taken

    def __init__(self, is_debug: bool):
        logging_level = logging.INFO if not is_debug else logging.DEBUG
        logging.basicConfig(level=logging_level,
                            format='[%(asctime)s] [%(threadName)-10s %(levelname)6s] --- %(message)s')
        self.begin_time = 0
        self.end_time = 0
        self.name = ""
        self.label_begin_time = 0
        self.label_name = ""
        self.time_profile_logger = logging.getLogger('Time profile')
        self.label_data = []

    def begin(self, name: str) -> None:
        self.begin_time = time.time()
        self.name = name

    def end(self) -> None:
        self.end_time = time.time()
        time_taken = self.end_time - self.begin_time
        if time_taken == 0:
            return
        self.time_profile_logger.debug("Time Profile: %-25s time: %fs", self.name, time_taken)

        for label in self.label_data:
            self.time_profile_logger.debug(" %-25s time: %f%%", label.name, label.time_taken / time_taken)
        self.label_data.clear()

    def label_begin(self, name: str) -> None:
        self.label_begin_time = time.time()
        self.label_name = name

    def label_end(self) -> None:
        label_end_time = time.time()
        time_taken = label_end_time - self.label_begin_time
        self.label_data.append(self.LabelData(self.label_name, time_taken))
