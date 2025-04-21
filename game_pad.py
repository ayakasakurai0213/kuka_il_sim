import pygame
from pygame.locals import *
import time


class Gamepad:
    def __init__(self, gamepad="joycon"):
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        pygame.init() 
        self.grip = -1
        
        if gamepad == "joycon":
            self.button = {"down": 1, "up": 2, "left": 9, "right": 10, "grip": 6}
        elif gamepad == "dualshock":
            self.button = {"down": 0, "up": 2, "left": 4, "right": 5, "grip": 10}
        
    def control(self):
        dx, dy, dz, da = 0, 0, 0, 0
        event_list = pygame.event.get()
        for e in event_list:
            if e.type == QUIT:
                return
            if e.type == pygame.locals.JOYAXISMOTION:
                dx = round(self.joystick.get_axis(0) / 130, 3) * -1
                dy = round(self.joystick.get_axis(1) / 130, 3)
                
                print(f"axis x: {dx}, axis y: {dy}")
                
        if self.joystick.get_button(self.button["down"]):
            dz -= 0.003
            print("down")
        if self.joystick.get_button(self.button["up"]):
            dz += 0.003
            print("up")
        if self.joystick.get_button(self.button["left"]):
            da += 0.10
            print("left")
        if self.joystick.get_button(self.button["right"]):
            da -= 0.10
            print("right")
        if self.joystick.get_button(self.button["grip"]):
            time.sleep(0.2)
            self.grip *= -1
            print("grip")
        return dx, dy, dz, da, self.grip
    
    
    def test_button(self):
        event_list = pygame.event.get()
        for e in event_list:
            if e.type == QUIT:
                return
        for i in range(self.joystick.get_numbuttons()):
            if self.joystick.get_button(i):
                print(f"button_idx: {i}")


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
        print("cannot find a joystick")