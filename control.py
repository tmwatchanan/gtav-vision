import random

from directkeys import PressKey, ReleaseKey, W, A, S, D

def control_decision(max_class):
    if max_class == 0:
        straight()
        return "straight"
    elif max_class == 1:
        reverse()
        return "reverse"
    elif max_class == 2:
        left()
        return "left"
    elif max_class == 3:
        right()
        return "right"
    elif max_class == 4:
        forward_left()
        return "forward+left"
    elif max_class == 5:
        forward_right()
        return "forward+right"
    elif max_class == 6:
        reverse_left()
        return "reverse+left"
    elif max_class == 7:
        reverse_right()
        return "reverse+right"
    else:
        no_keys()
        return "nokeys"

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    #ReleaseKey(S)

def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)