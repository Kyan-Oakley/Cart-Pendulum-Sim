from cmu_graphics import *
import numpy as np
import math

PPM = 100 #Pixels Per Meter

################################################################APP START FUNCTIONS################################################################
def onAppStart(app):
    #Resizing Window and timing
    app.height = 800
    app.width = 1600
    app.stepspersecond = 60
    app.startScreen = True

    #Some constants we will use (and be able to change)
    app.length = 2
    app.cartMass = 10
    app.pendMass = 1
    app.gravity = 9.8

    resetSim(app)

    app.labels = [["Cart Mass (kg)", "cartMass", 1], ["Pendulum Mass (kg)", "pendMass", 1], ["Pendulum Length (m)", "length", 0.25], ["Gravitational Constant (m/s^2)", "gravity", 0.1]]
    app.selectedIndex = None

def resetSim(app):
    #Force on cart initially set to zero
    app.Q = 0

    #Position of cart initially in middle and set in meters (to make physics easier) (posY will not change so it will not be changed or present in the state space)
    app.posY = app.height / (2 * PPM)
    app.posX = app.width / (2 * PPM)

    #Other parameters
    app.xDot = 0
    app.xDubDot = 0
    app.xPrevDubDot = 0
    app.theta = math.pi / 12
    app.thetaDot = 0
    app.thetaDubDot = 0
    app.thetaPrevDubDot = 0

    #Create initial accelerations
    verletAccUpdate(app)

#################################################################DRAWING FUNCTIONS#################################################################
def redrawAll(app):
    if app.startScreen:
        #Allow options for changing constants of simulation
        drawStartScreen(app)

    else:  
        drawSim(app)


def drawStartScreen(app):
    inc = app.width / (len(app.labels) + 1)
    for i in range(len(app.labels)):
        title, attrName, _ = app.labels[i]
        xPos = inc * (i + 1)
        if i == app.selectedIndex:
            color = "crimson"
        else:
            color = "lightBlue"
        drawCircle(xPos, app.posY * PPM, 50, border = "black", fill = color)
        drawLabel(f"{title}", xPos, app.posY * PPM - 75, size = 16, bold = True)
        drawLabel(f"{getattr(app, attrName)}", xPos, app.posY * PPM, size = 32, bold = True)


def drawSim(app):
    #Finding endpoint of pendulum
    endpoint = (app.posX + app.length * np.sin(app.theta), app.posY - app.length * np.cos(app.theta))

    #Drawing cart-pendulum system
    drawRect(app.posX * PPM - 50, app.posY * PPM - 25, 100, 50)
    drawLine(app.posX * PPM, app.posY * PPM, endpoint[0] * PPM, endpoint[1] * PPM)
    drawCircle(endpoint[0] * PPM, endpoint[1] * PPM, 25)
    drawCircle(app.posX * PPM - 35, app.posY * PPM + 30, 10)
    drawCircle(app.posX * PPM + 35, app.posY * PPM + 30, 10)
    drawLine(0, app.posY * PPM + 40, app.width, app.posY * PPM + 40)

###################################################################UX FUNCTIONS###################################################################
def onKeyPress(app, key):
    if key == "s":
        app.startScreen = not app.startScreen
        resetSim(app)
    if app.selectedIndex != None:
        attrName, step = app.labels[app.selectedIndex][1], app.labels[app.selectedIndex][2]
        if key == "up":
            setattr(app, attrName, getattr(app, attrName) + step)
        if key == "down":
            setattr(app, attrName, getattr(app, attrName) - step)

def onKeyHold(app, keys):
    if not app.startScreen:
        if "right" in keys:
            app.Q = 200
        elif "left" in keys:
            app.Q = -200
    else:
        pass

    
def onkeyRelease(app, key):
    if not app.startScreen:
        if key == "right" or key == "left":
            app.Q = 0

def onMousePress(app, mouseX, mouseY):
    inc = app.width / (len(app.labels) + 1)
    happened = False
    for i in range(len(app.labels)):
        xPos = inc * (i + 1)
        if distance(xPos, app.posY * PPM, mouseX, mouseY) <= 100:
            app.selectedIndex = i
            happened = True
    if not happened:
        app.selectedIndex = None
    

def distance(x0, y0, x1, y1):
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    
##########################################################INTEGRATOR/CONTROLLER FUNCTIONS#########################################################
def onStep(app):
    #Verlet integration being used because total mechanical integration of the system must be preserved
    verlet(app)

def verlet(app):
    #Using Stormer Verlet integration method because the mechanical energy needs to be preserved
    Verlet(app)

def Verlet(app):
    #Update positions based on old accels, solve new accels, update velocity
    verletPosUpdate(app)
    verletAccUpdate(app)
    verletVelUpdate(app)

def verletPosUpdate(app):
    #Numerically solve for new positions
    xFirstOrderTerm = app.posX + app.xDot * (1 / app.stepspersecond)
    xSecondOrderTerm = 0.5 * app.xDubDot * (1 / app.stepspersecond) ** 2
    thetaFirstOrderTerm = app.theta + app.thetaDot * (1 / app.stepspersecond)
    thetaSecondOrderTerm = 0.5 * app.thetaDubDot * (1 / app.stepspersecond) ** 2
    app.posX = xFirstOrderTerm + xSecondOrderTerm
    app.theta = thetaFirstOrderTerm + thetaSecondOrderTerm

def verletAccUpdate(app):
    #Solve giant matrix system for new accelerations
    app.xPrevDubDot = app.xDubDot
    app.thetaPrevDubDot = app.thetaDubDot
    stateMatrix = np.array([[np.cos(app.theta), app.length], [app.cartMass + app.pendMass, app.pendMass * app.length * np.cos(app.theta)]], dtype = float)
    transformedPoint = np.array([app.gravity * np.sin(app.theta), app.Q + app.pendMass * app.length * (app.thetaDot ** 2) * np.sin(app.theta)], dtype = float)
    solution = np.linalg.solve(stateMatrix, transformedPoint)
    app.xDubDot = solution[0]
    app.thetaDubDot = solution[1]

def verletVelUpdate(app):
    #Update velocities based on the new accelerations and the old ones
    app.xDot += 0.5 * (app.xPrevDubDot + app.xDubDot) * (1 / app.stepspersecond)
    app.thetaDot += 0.5 * (app.thetaPrevDubDot + app.thetaDubDot) * (1 / app.stepspersecond)



runApp()