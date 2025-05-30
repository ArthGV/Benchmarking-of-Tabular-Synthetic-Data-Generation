from tqdm import tqdm

class TqdmTracker:
    def __init__(self, total=None):
        self.pbar = tqdm(total=total)
    def update(self, n):
        self.pbar.update(n)
    def close(self):
        self.pbar.close()