from base_gym_class import GymAtari, Wrapper
from derived_gym_class import RepeatAction, PreProcess, Stack
def make(start_value):
    zero = GymAtari(start_value)
    zero = Wrapper(zero)    # Wrapper class contains object of GymAtari as attribute named env
    return zero

zero = make(0)
zero= RepeatAction(zero, 3)  # RepeatAction class contains object of Wrapper as attribute named env
zero= PreProcess(zero)     # PreProcess class contains object of RepeatAction as attribute named env
zero= Stack(zero)         # Stack class contains object of PreProcess class as attribute named env
a= zero.step(-1)           # zero is an object of Stack class
print(a)              
                            