from msgspec.json import Encoder
from hashlib import md5


class JsonObjHasher:
    def __init__(self):
        self.encoder = Encoder()

    def __call__(self, obj) -> str:
        return md5(self.encoder.encode(obj)).hexdigest()
