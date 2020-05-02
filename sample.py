import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def start(mode, to_recv, to_send):
    
    while True: 
        obs = to_recv.get()
        if obs == None:
            break
        action = random.randint(0,2)
        to_send.put(action)