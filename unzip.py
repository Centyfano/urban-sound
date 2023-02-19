import os
import gzip

class UrbanSoundZipped:
    def __init__(self, zipped_path) -> None:
        self.zipped_path = os.path.abspath(zipped_path)
        self.path = os.path.abspath(self.zipped_path)
        self.chunk_size = 1024 * 1024 * 500
    
    def extract(self):
        _dir = self.zipped_path.split('.tar.gz')[0]
        if not os.path.isdir(_dir):
            with gzip.open(self.zipped_path, 'rb') as f_in, open(self.zipped_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(self.chunk_size)
                    if not chunk:
                        break
                    f_out.write(self.chunk_size)
