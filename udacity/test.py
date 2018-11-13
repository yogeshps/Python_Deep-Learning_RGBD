import datetime

class Human(object):
    name = None
    gender = None
    bday = None

    def __getattr__(self, name):
        if name == 'age':
            print (datetime.datetime.now() - self.bday)
        else:
            return None
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

h = Human()
h.bday = datetime.datetime(1984,8,20)
h.age = 25
print (h.age)