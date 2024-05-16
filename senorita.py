#example of class
class area():
    def __init__(self, l):
        self.l = l

    def square(self):
        return self.l * self.l


david = area(5)
print(david.square())

#example of inheritance of prev class
class volume(area):
    def __init__(self, l, h):
        self.l = l
        self.h = h

    def vol(self):
        return self.l*self.l*self.h

john = volume(6, 2)
print(john.square(), john.vol())
