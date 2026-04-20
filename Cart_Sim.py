from cmu_graphics import *
import numpy as np
from scipy.linalg import schur
import math

PPM = 400 #Pixels Per Meter

################################################################APP START FUNCTIONS################################################################
def onAppStart(app):
    #Resizing Window and timing
    app.height = 800
    app.width = 1600
    app.stepspersecond = 60
    app.startScreen = True

    #Allows for controller selection
    app.controlsPage = False
    app.controllerSelect = None
    app.controllerEnabled = True
    app.controllers = {
        "Proportional-Integral-Derivative" : {
            "Proportional Term" : {"attr": "ktp", "step": 1},
            "Integral Term"     : {"attr": "kti", "step": 0.01},
            "Derivative Term"   : {"attr": "ktd", "step": 1},
        },
        "Linear Quadratic Regulator" : {
            "Position Term" : {"attr" : "posPunish", "step" : 0.5},
            "Velocity Term" : {"attr" : "velPunish", "step" : 0.5},
            "Angle Term" : {"attr" : "thetaPunish", "step" : 1},
            "Angular Velocity Term": {"attr" : "thetaVelPunish", "step" : 1}
        }
    }
    app.paramSelected = None

    #Some constants we will use (and be able to change)
    app.length = 0.5
    app.cartMass = 10
    app.pendMass = 1
    app.gravity = 9.8

    resetSim(app)

    app.labels = [["Cart Mass (kg)", "cartMass", 1],
                  ["Pendulum Mass (kg)", "pendMass", 1],
                  ["Pendulum Length (m)", "length", 0.1],
                  ["Gravitational Constant (m/s^2)", "gravity", 0.1]]
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
    app.theta = 0.55
    app.thetaDot = 0
    app.thetaDubDot = 0
    app.thetaPrevDubDot = 0

    #Controller reset values
    app.PIDTotal = 0
    app.PIDError = [0, 0]

    #Determines constants for PID controller
    app.ktp = 200
    app. kti = 0.1
    app.ktd = 35

    #Determines constants for LQR and LQG controllers
    app.posPunish = 10
    app.velPunish = 2
    app.thetaPunish = 150
    app.thetaVelPunish = 30
    app.controlPunish = 0.01
    app.firstRun = True

    #Create initial accelerations
    verletAccUpdate(app)

#################################################################DRAWING FUNCTIONS#################################################################
def redrawAll(app):
    if app.startScreen and not app.controlsPage:
        drawStartScreen(app)

    elif app.startScreen and app.controlsPage:
        drawControllerScreen(app)

    else:  
        drawSim(app)

def drawStartScreen(app):
    drawLabel("Press 'S' to start the simulator", app.width/2, 50, size = 32, bold = True)
    drawLabel("Press 'P' to open the controller selector", app.width/2, 80, size = 32, bold = True)
    controllerBoxColor = "white" if app.controllerSelect == None else "lightGreen"
    drawRect(app.width/2, 130, 400, 40,  fill = controllerBoxColor, border = "Black", align = "center")
    drawLabel(f"Controller Selected: {app.controllerSelect}", app.width/2, 130, size = 16, bold = True)
    inc = app.width / (len(app.labels) + 1)
    for i in range(len(app.labels)):
        title, attrName, _ = app.labels[i]
        xPos = inc * (i + 1)
        if i == app.selectedIndex:
            color = "crimson"
        else:
            color = "lightBlue"
        drawCircle(xPos, app.posY * PPM, 60, border = "black", fill = color)
        drawLabel(f"{title}", xPos, app.posY * PPM - 75, size = 16, bold = True)
        drawLabel(f"{rounded(100 * getattr(app, attrName)) / 100}", xPos, app.posY * PPM, size = 32, bold = True)

def drawControllerScreen(app):
    drawLabel("Press 'S' to start the simulator", app.width/2, 50, size = 32, bold = True)
    drawLabel("Press 'P' to return to the system parameters selector", app.width/2, 80, size = 32, bold = True)
    incX = app.width / (len(app.controllers) + 1)
    incY = 110
    for i, controller in enumerate(list(app.controllers)):
        posX = incX * (i + 1)
        titleColor = "crimson" if controller == app.controllerSelect else "black"
        drawRect(posX - (incX // 2 - 20), 180, incX - 40, 600, fill = "white", border = "black")
        drawLabel(f"{controller}", posX, 200, size = 24, bold = True, fill = titleColor)
        for j, paramName in enumerate(list(app.controllers[controller])):
            paramInfo = app.controllers[controller][paramName]
            if j == app.paramSelected and controller == app.controllerSelect:
                color = "crimson"
            else:
                color = "lightBlue"
            posY = 200 + incY * (j + 1)
            drawCircle(posX + 70, posY, 50, border = "black", fill = color)
            drawLabel(f"{paramName}", posX - 70, posY, size = 14, bold = True)
            drawLabel(f"{rounded(100 * getattr(app, paramInfo['attr'])) / 100}", posX + 70, posY, size = 32, bold = True)


def drawSim(app):
    drawLabel("Press 'S' to return to the system parameters selector", app.width/2, 50, size = 32, bold = True)
    drawLabel("Press 'R' to recenter the cart", app.width/2, 80, size = 32, bold = True)
    if app.controllerSelect != None:
        drawLabel("Press 'P' to toggle the selected controller", app.width/2, 110, size = 32, bold = True)
    status_box = (20, 20, 220, 50)  # x, y, w, h
    if app.controllerSelect != None:
        bx, by, bw, bh = status_box
        fillColor = "lightGreen" if app.controllerEnabled else "lightCoral"
        drawRect(bx, by, bw, bh, fill = fillColor, border = "black", borderWidth = 2)
        status = "ON" if app.controllerEnabled else "OFF"
        drawLabel(f"Controller: {status}", bx + bw / 2, by + bh / 2, size = 20, bold = True)
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
        solveForKMatrix(app)
    if app.startScreen:
        if key == "p":
            app.controlsPage = not app.controlsPage
        elif app.controlsPage:
            if key == "up" or key == "down":
                if app.paramSelected is not None and app.controllerSelect is not None:
                    paramName = list(app.controllers[app.controllerSelect])[app.paramSelected]
                    paramInfo = app.controllers[app.controllerSelect][paramName]
                    delta = paramInfo["step"] if key == "up" else -paramInfo["step"]
                    setattr(app, paramInfo["attr"], getattr(app, paramInfo["attr"]) + delta)
        else:
            if key == "up" and app.selectedIndex is not None:
                attrName, step = app.labels[app.selectedIndex][1], app.labels[app.selectedIndex][2]
                setattr(app, attrName, getattr(app, attrName) + step)
            elif key == "down" and app.selectedIndex is not None:
                attrName, step = app.labels[app.selectedIndex][1], app.labels[app.selectedIndex][2]
                setattr(app, attrName, getattr(app, attrName) - step)
    else:
        if (key == "p") and (app.controllerSelect != None):
            app.controllerEnabled = not app.controllerEnabled
            if app.controllerEnabled:
                # Reset integral windup and prime error so derivative starts at 0
                app.PIDTotal = 0
                app.PIDError = [app.measuredTheta, app.measuredVel]
        elif key == "r":
            app.posX = app.width / (2 * PPM)
            

def onKeyHold(app, keys):
    if not app.startScreen:
        if "right" in keys:
            app.Q = 200
        elif "left" in keys:
            app.Q = -200
    else:
        pass

    
def onKeyRelease(app, key):
    if not app.startScreen:
        if key == "right" or key == "left":
            app.Q = 0

def onMousePress(app, mouseX, mouseY):
    if app.startScreen and app.controlsPage:
        incX = app.width / (len(app.controllers) + 1)
        incY = 110
        happened = False
        for i, controller in enumerate(list(app.controllers)):
            posX = incX * (i + 1)
            bx = posX - (incX // 2 - 20)
            by = 180
            bw = incX - 40
            bh = 600
            if bx <= mouseX <= bx + bw and by <= mouseY <= by + bh:
                app.controllerSelect = controller
                happened = True
                paramHit = False
                for j, paramName in enumerate(list(app.controllers[controller])):
                    posY = 200 + incY * (j + 1)
                    if distance(posX + 70, posY, mouseX, mouseY) <= 70:
                        app.paramSelected = j
                        paramHit = True
                        break
                if not paramHit:
                    app.paramSelected = None
                break
        if not happened:
            app.paramSelected = None
            app.controllerSelect = None
    elif app.startScreen:
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
    #Put theta between -pi and pi
    rescaleTheta(app)

    #Verlet integration being used because total mechanical integration of the system must be preserved
    verlet(app)

    #Start running controller select
    app.measuredTheta = app.theta
    app.measuredVel = app.xDot
    if app.controllerSelect == "Proportional-Integral-Derivative" and app.controllerEnabled == True and not app.startScreen:
        PID(app)
    elif app.controllerSelect == "Linear Quadratic Gaussian" and app.controllerEnabled == True and not app.startScreen:
        if app.firstRun == True:
            solveForKMatrix(app)
            app.firstRun = False
        LQG(app)
    elif app.controllerSelect == "Linear Quadratic Regulator" and app.controllerEnabled == True and not app.startScreen:
        if app.firstRun == True:
            solveForKMatrix(app)
            app.firstRun = False
        LQR(app)
    
def rescaleTheta(app):
    app.theta = (app.theta % (2 * math.pi))
    if app.theta > math.pi:
        app.theta -= 2 * math.pi

def verlet(app):
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

def PID(app):
    #Performs PID (Proportional-Integral-Derivative) control on the angle of the 

    if app.measuredTheta >= -0.65 and app.measuredTheta < 0.65:
        #Determines constants for controller
        kxp = 6
        desiredAngle = 0
        desiredVel = 0

        #Builds matrix and vector for PID
        prevError = app.PIDError.copy()
        app.PIDError[0] = app.measuredTheta - desiredAngle
        app.PIDError[1] = app.measuredVel - desiredVel
        app.PIDTotal += app.PIDError[0]
        errorTDot = app.stepspersecond * (app.PIDError[0] - prevError[0])

        state = np.array([app.PIDError[0], errorTDot, app.PIDError[1], app.PIDTotal]).T
        matrix = np.array([app.ktp, app.ktd, kxp, app.kti])

        #Computes PID
        app.Q = matrix @ state


    else:
        swingUp(app)

def swingUp(app):
    #Find total energy of system and energy error
    pendulumKineticEnergy = 0.5 * app.pendMass * (app.length * app.thetaDot) ** 2
    pendulumPotentialEnergy = app.pendMass * app.gravity * app.length * np.cos(app.theta)
    totalPendulumEnergy = pendulumKineticEnergy + pendulumPotentialEnergy

    desiredPendulumEnergy = app.pendMass * app.gravity * app.length
    pendulumEnergyError = desiredPendulumEnergy - totalPendulumEnergy

    #Tuning gain
    k = -3
    kx = 0.2
    kxd = 2 * (kx * app.cartMass) ** 0.5 + 0

    forceOnPendulum = k * app.thetaDot * np.cos(app.theta) * pendulumEnergyError - kx * (app.posX - app.width / (2 * PPM)) - kxd * app.xDot

    app.Q = forceOnPendulum

    #Reset integral and derivative terms for PID controller to avoid windup
    app.PIDTotal = 0
    app.PIDError = [app.measuredTheta, app.measuredVel]

def solveForKMatrix(app):
    #Solves Algebraic Riccati Equation to determine the optimal K gain matrix given the constraint function

    #Create system dynamics matricies
    a = -(app.pendMass * app.gravity / app.cartMass)
    b = app.gravity * (app.pendMass + app.cartMass) / (app.length * app.cartMass)

    jacobian = np.array([[0, 1, 0, 0],
                         [0, 0, a, 0],
                         [0, 0, 0, 1],
                         [0, 0, b, 0]])
    
    controllerCost = np.array([0, app.cartMass ** (-1), 0, -(app.cartMass * app.length) ** (-1)])

    #Create LQR optimization matricies
    Q = np.diag([app.posPunish, app.velPunish, app.thetaPunish, app.thetaVelPunish])
    R = app.controlPunish

    #Create hamiltonian matrix to decompose into optimal solutions
    upperRight = np.outer(controllerCost, controllerCost) / R
    H = np.block([[jacobian, -1 * upperRight],
                  [-1 * Q, -1 * jacobian.T]])
    
    n = jacobian.shape[0]
    _, Z, sdim = schur(H, sort="lhp")
    assert(sdim == n)

    Z11 = Z[:n, :n]
    Z21 = Z[n:, :n]

    #Compute P matrix and use to solve K
    P = Z21 @ np.linalg.inv(Z11)
    app.KMatrix = (1/R) * (controllerCost.T @ P)

def LQG(app):
    angle = kalmanFilter(app.measuredTheta)
    LQR(app)

def kalmanFilter(measuredTheta):
    #Filters out error based on system dynamics
    pass

def LQR(app):
    #Performs Linear Quadratic Regulation control based on the already discovered K matrix
    if app.theta >= -0.65 and app.theta <= 0.65:
        #Defines target values and their errors
        desiredPos = app.width / (2 * PPM)
        desiredVel = 0
        desiredTheta = 0
        desiredThetaVel = 0

        posError = app.posX - desiredPos
        velError = app.xDot - desiredVel
        thetaError = app.theta - desiredTheta
        thetaVelError = app.thetaDot - desiredThetaVel

        #Creates state vector then solve for control output
        state = np.array([posError, velError, thetaError, thetaVelError]).T
        app.Q = -(app.KMatrix @ state)
    else:
        swingUp(app)


runApp()