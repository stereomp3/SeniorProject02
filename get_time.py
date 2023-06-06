from datetime import datetime

def get_time():
    time_now = datetime.now()
    # current_time = time_now.strftime("%R")
    current_time = time_now.strftime("%r")
    h, m, s = current_time.split(":")
    return h, m, s[-2:]