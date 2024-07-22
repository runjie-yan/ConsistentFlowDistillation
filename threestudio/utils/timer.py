import threestudio
from time import time
from threestudio.utils.typing import *
class context_timer:
    def __init__(self, tag):
        self.tag = tag
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        end_time = time()
        threestudio.info(f"{self.tag}: {end_time-self.start_time:.5f}s")
        

class freq_timer:
    def __init__(self, eps=1e-8) -> None:
        self.cur_time = time()
        self.eps = eps
        
    def get_freq(self) -> float:
        last_cur_time = self.cur_time
        self.cur_time = time()
        return 1./(self.cur_time-last_cur_time+self.eps)