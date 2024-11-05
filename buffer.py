
class Buffer:   
    def __init__(self, size: int):
        self.size = size

class Base(Buffer):
    def __init__(self, size: int):
        super().__init__(size)
        self.address = None # todo
        
    def get_global_offset():
        return 0

class View(Buffer):
    def __init__(self, size: int, offset: int, buffer: Buffer):
        super().__init__(size)
        self.offset: int = offset
        self.buffer: Buffer = buffer
        
    def get_global_offset(self):
        return self.buffer.get_global_offset() + self.offset