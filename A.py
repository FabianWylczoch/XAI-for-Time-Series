class outerA ():
    def __init__(self):
        print("init outerA")

class A (outerA):
    def __init__(self):
        super().__init__()
        print("init A")
