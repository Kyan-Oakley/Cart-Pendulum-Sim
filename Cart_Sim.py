from cmu_graphics import *
import numpy as np

PPM = 100 #Pixels Per Meter

#Physical object class for physics simulation everything in SI units, only converted to pixels after
class Mass:

    #Initializer holding all the necessary physical properties of the mass
    def __init__(self, mass):
        self.mass = mass
        self.pos = (0, 0)
        self.vel = (0, 0)
        self.acc = (0, 0)
    
    #applyForce method used to apply a force
    def applyForce(self, force):
        ax = force[0] / self.mass
        ay = force[1] / self.mass
        self.acc = (ax, ay)
    
    def constrain(self, xCon, yCon):
        pass

cart = Mass(1)

def onAppStart(app):
    app.height = 800
    app.width = 800
    app.stepspersecond = 60
    cart.pos = (app.width / (2 * PPM), app.height / (2 * PPM))


def redrawAll(app):
    #Draw cart
    drawRect(PPM * cart.pos[0] - 50, PPM * cart.pos[1] - 25, 100, 50, fill='black')

def onKeyPress(app, key):
    if key == 'right':
        cart.applyForce((10, 0))
    elif key == 'left':
        cart.applyForce((-10, 0))
    
def onStep(app):
    integrateCart(app)

#Integrates cart position over time
def integrateCart(app):
    cart.pos = (cart.pos[0] + cart.vel[0]/app.stepspersecond, cart.pos[1] + cart.vel[1]/app.stepspersecond)
    cart.vel = (cart.vel[0] + cart.acc[0]/app.stepspersecond, cart.vel[1] + cart.acc[1]/app.stepspersecond)
    cart.acc = (0, 0)



runApp()