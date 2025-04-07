import pygame
from pygame.locals import *
import time


class Gamepad:
    def __init__(self):
        pygame.joystick.init()
        self.joycon = pygame.joystick.Joystick(0)
        self.joycon.init()
        pygame.init() 
        self.grip = -1
        
    def control(self):
        dx, dy, dz, da = 0, 0, 0, 0
        event_list = pygame.event.get()
        for e in event_list:
            if e.type == QUIT:
                return
            if e.type == pygame.locals.JOYAXISMOTION:
                dx = round(self.joycon.get_axis(0) / 130, 3) * -1
                dy = round(self.joycon.get_axis(1) / 130, 3)
                
                print(f"axis x: {dx}, axis y: {dy}")
                
        if self.joycon.get_button(1):
            dz = -0.003
            print("down")
        if self.joycon.get_button(2):
            dz = 0.003
            print("up")
        if self.joycon.get_button(9):
            da = 0.10
            print("left")
        if self.joycon.get_button(10):
            da = -0.10
            print("right")
        if self.joycon.get_button(6):
            time.sleep(0.2)
            self.grip *= -1
            print("grip")
        return dx, dy, dz, da


def main():
    gamepad = Gamepad()
    print(pygame.joystick.get_count())
    print("joystick start") 
    
    while True:
        # print(gamepad.grip)
        gamepad.control()     
            
if __name__ == "__main__":
    try:
        main()
    except pygame.error:
        print("cannot find a joycon")