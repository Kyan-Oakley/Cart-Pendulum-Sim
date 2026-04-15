from cmu_graphics import *
import numpy as np
import math

PPM = 100 #Pixels Per Meter


def onAppStart(app):
    #Resizing Window and timing
    app.height = 800
    app.width = 800
    app.stepspersecond = 60

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

    #Some constants we will use (and be able to change)
    app.length = 2
    app.cartMass = 2
    app.pendMass = 1
    app.gravity = 9.81

    #Create initial accelerations
    verletAccUpdate(app)


def redrawAll(app):
    #Finding endpoint of pendulum
    endpoint = (app.posX + app.length * np.sin(app.theta), app.posY - app.length * np.cos(app.theta))

    #Drawing cart-pendulum system
    drawRect(app.posX * PPM - 50, app.posY * PPM - 25, 100, 50)
    drawLine(app.posX * PPM, app.posY * PPM, endpoint[0] * PPM, endpoint[1] * PPM)
    drawCircle(endpoint[0] * PPM, endpoint[1] * PPM, 25)
    drawCircle(app.posX * PPM - 35, app.posY * PPM + 30, 10)
    drawCircle(app.posX * PPM + 35, app.posY * PPM + 30, 10)
    drawLine(0, app.posY * PPM + 40, app.width, app.posY * PPM + 40)

def onKeyHold(app, keys):
    if "right" in keys:
        app.Q = 40
    elif "left" in keys:
        app.Q = -40
    
def onStep(app):
    verletPosUpdate(app)
    verletAccUpdate(app)
    verletVelUpdate(app)

def verletPosUpdate(app):
    #Verlet integration being used to perserve mechanical energy of the system, first and second order terms are used to numerically integrate
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
    #Update velocities based on the new accelerations
    app.xDot += 0.5 * (app.xPrevDubDot + app.xDubDot) * (1 / app.stepspersecond)
    app.thetaDot += 0.5 * (app.thetaPrevDubDot + app.thetaDubDot) * (1 / app.stepspersecond)



runApp()