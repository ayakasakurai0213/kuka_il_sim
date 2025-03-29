import pygame
from pygame.locals import *
import time

import pygame.locals

def main():
    pygame.joystick.init()
    print(pygame.joystick.get_count())
    joycon_l = pygame.joystick.Joystick(0)
    joycon_l.init()

    print("joystick start")

    pygame.init()

    while True:
        event_list = pygame.event.get()
        
        for e in event_list:
            if e.type == QUIT:
                return
            
            if e.type == pygame.locals.JOYAXISMOTION:
                x, y = joycon_l.get_axis(0), joycon_l.get_axis(1)
                print(f"axis x: {x}, axis y: {y}")
            elif e.type == pygame.locals.JOYAXISMOTION:
                x, y = joycon_l.get_hat(0)
                print(f"hat x: {x}, hat y: {y}")
            elif e.type == pygame.locals.JOYBUTTONDOWN:
                print(f"button: {e.button}")
                
        time.sleep(0.1)
            
if __name__ == "__main__":
    # try:
        main()
    # except pygame.error:
    #     print("cannot find a joycon")