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
    app.infoScreen = False

    #Allows for controller selection
    app.controlsPage = False
    app.controllerSelect = None
    app.controllerEnabled = True
    app.controllers = {
        "Proportional-Integral-Derivative" : {
            "Proportional Term" : {"attr": "ktp", "step": 10},
            "Integral Term"     : {"attr": "kti", "step": 0.01},
            "Derivative Term"   : {"attr": "ktd", "step": 1},
        },
        "Linear Quadratic Regulator" : {
            "Position Term" : {"attr" : "posPunish", "step" : 0.5},
            "Velocity Term" : {"attr" : "velPunish", "step" : 0.5},
            "Angle Term" : {"attr" : "thetaPunish", "step" : 1},
            "Angular-Velocity Term": {"attr" : "thetaVelPunish", "step" : 1}
        },
        "Linear Quadratic Gaussian" : {
            "Position Term" : {"attr" : "posPunish", "step" : 0.5},
            "Velocity Term" : {"attr" : "velPunish", "step" : 0.5},
            "Angle Term" : {"attr" : "thetaPunish", "step" : 10},
            "Angular-Velocity Term": {"attr" : "thetaVelPunish", "step" : 5}
        }
    }
    app.paramSelected = None

    #Stuff for live toggling
    app.liveSelectedControllerAttribute = None
    app.noiseSelected = False
    app.noiseLevel = 1
    app.liveSelectedSystemParam = None

    #Some constants we will use (and be able to change)
    app.length = 0.5
    app.cartMass = 10
    app.pendMass = 1
    app.gravity = 9.8

    #Determines constants for PID controller
    app.ktp = 200
    app. kti = 0.1
    app.ktd = 35

    #Determines constants for LQR and LQG controllers
    app.posPunish = 10
    app.velPunish = 2
    app.thetaPunish = 150
    app.thetaVelPunish = 40
    app.controlPunish = 0.01

    #Sets and resets the conditions of the simulation
    resetSim(app)

    #Sets constants for Gaussian noise
    setGaussianConstants(app)

    app.labels = [["Cart Mass (kg)", "cartMass", 1],
                  ["Pendulum Mass (kg)", "pendMass", 1],
                  ["Pendulum Length (m)", "length", 0.1],
                  ["Gravity Constant (m/s^2)", "gravity", 0.1]]
    app.selectedIndex = None

def setGaussianConstants(app):
    app.gaussianMean = 0
    app.gaussianSTDPos = 0.05
    app.gaussianSTDVel = 0.05
    app.gaussianSTDTheta = 0.025
    app.gaussianSTDThetaVel = 0.025

def resetSim(app):
    #Force on cart initially set to zero
    app.perturbQ = 0
    app.controllerQ = 0
    app.Q = 0

    #Position of cart initially in middle and set in meters (to make physics easier) (posY will not change so it will not be changed or present in the state space)
    app.posY = app.height / (2 * PPM)
    app.posX = app.width / (2 * PPM)

    #Other parameters
    app.xDot = 0
    app.xDubDot = 0
    app.xPrevDubDot = 0
    app.theta = -3
    app.thetaDot = 0
    app.thetaDubDot = 0
    app.thetaPrevDubDot = 0

    #Controller reset values
    app.PIDTotal = 0
    app.PIDError = [0, 0]

    #EKF state estimate and covariance
    app.ekf_state = np.array([app.posX, 0.0, app.theta, 0.0])
    app.ekf_P = np.eye(4)

    #Reset work done by controller
    app.workDone = 0

    #Create initial accelerations
    verletAccUpdate(app)

    #Resolve for the K matrix
    solveForKMatrix(app)

#################################################################DRAWING FUNCTIONS#################################################################
def redrawAll(app):
    if app.startScreen and not app.controlsPage:
        #Draws the start screen to control the parameters
        drawStartScreen(app)

    elif app.startScreen and app.controlsPage and app.infoScreen:
        drawImage("controllerInformation.png", 0, 0, width=app.width, height=app.height)
        drawRect(-5, -5, 100, 100, border = "black", fill = "lightGray")
        drawImage("houseIcon.png", 0, 0, width=90, height=90)

    elif app.startScreen and app.controlsPage:
        #Draws the controller selection and tuning screen
        drawControllerScreen(app)

    else:  
        #Draws the simulator
        drawSim(app)

def drawStartScreen(app):
    #Draw keybinds and controller box
    drawLabel("Press 'S' to start the simulator", app.width/2, 50, size = 32, bold = True)
    drawLabel("Press 'P' to open the controller selector", app.width/2, 80, size = 32, bold = True)
    controllerBoxColor = "white" if app.controllerSelect == None else "lightGreen"
    drawRect(app.width/2, 130, 400, 40,  fill = controllerBoxColor, border = "Black", align = "center")
    drawLabel(f"Controller Selected: {app.controllerSelect}", app.width/2, 130, size = 16, bold = True)

    #Draws the system parameter buttons
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

    #Draw toggle for info screen
    drawRect(-5, -5, 100, 100, fill = "lightGray", border = "black")
    drawImage("infoIcon.png", 0, 0, width=90, height=90)

    #Draws the controller boxes
    for i, controller in enumerate(list(app.controllers)):
        posX = incX * (i + 1)
        titleColor = "crimson" if controller == app.controllerSelect else "black"
        drawRect(posX - (incX // 2 - 20), 180, incX - 40, 600, fill = "white", border = "black")
        drawLabel(f"{controller}", posX, 200, size = 24, bold = True, fill = titleColor)
        #Draws the controler parameter buttons
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
    drawLabel(f"Work done by controller: {int(app.workDone) / 1000} kJ", app.width - 175, 25, size = 20)
    if app.controllerSelect != None:
        drawLabel("Press 'P' to toggle the selected controller", app.width/2, 110, size = 32, bold = True)
    #Draws controller status box
    drawControllerBox(app)

    #Draws controller selector box
    drawControllerSelectorBox(app)

    #Draw legend for the force arrows and their values
    drawForceBox(app)

    #Draws selection menu for controller attributes
    drawControllerAttributes(app)

    #Draws noise level box
    drawNoiseBox(app)

    #Draws selection menu for system parameters
    drawSystemParams(app)

    #Draws cart-pendulum system
    drawCartSys(app)

    #Draws force arrows
    drawForces(app)

def drawControllerBox(app):
    status_box = (20, 20, 300, 50)  # x, y, w, h
    if app.controllerSelect != None:
        bx, by, bw, bh = status_box
        fillColor = "lightGreen" if app.controllerEnabled else "lightCoral"
        drawRect(bx, by, bw, bh, fill = fillColor, border = "black", borderWidth = 2)
        status = "ON" if app.controllerEnabled else "OFF"
        drawLabel(f"{app.controllerSelect}: {status}", bx + bw / 2, by + bh / 2, size = 16, bold = True)

def drawControllerSelectorBox(app):
    startPos = (app.width - 125, 45)

    drawRect(startPos[0], startPos[1], 100, 90, fill = "lightGrey", opacity = 50, border = "black")
    labels = ["PID", "LQR", "LQG"]

    if app.controllerSelect == "Proportional-Integral-Derivative":
        selectedIndex = 0
    elif app.controllerSelect == "Linear Quadratic Regulator":
        selectedIndex = 1
    elif app.controllerSelect == "Linear Quadratic Gaussian":
        selectedIndex = 2
    else:
        selectedIndex = None

    inc = 30
    for i, label in enumerate(labels):
        if i == selectedIndex:
            color = "crimson"
        else:
            color = "black"
        posX, posY = app.width - 75, 60 + inc * i
        drawLabel(f"{label}", posX, posY, size = 20, fill = color)

def drawForceBox(app):
    if app.controllerSelect != None:
        startPos = (app.width - 300, app.height // 2 + 40)
        drawRect(startPos[0], startPos[1], 305, 100, fill = "lightGray", border = "black", opacity = 50)
        drawLabel(f"Controller Force: {rounded(100 * app.controllerQ) / 100}N", app.width - 150, app.height // 2 + 73, fill = "green", size = 24)
        drawLabel(f"External Force: {rounded(100 * app.perturbQ) / 100}N", app.width - 150, app.height // 2 + 107, fill = "red", size = 24)

def drawControllerAttributes(app):
    if app.controllerSelect != None:
        inc = 175
        width = 50 + len(app.controllers[app.controllerSelect]) * inc
        leftSide = app.width - width
        drawRect(leftSide, app.height - 200, width, 200, fill = "lightGrey", opacity = 50, border = "black")
        for i, att in enumerate(list(app.controllers[app.controllerSelect])):
            posY = app.height - 100
            posX = leftSide + (107.5 + i * inc)
            if app.liveSelectedControllerAttribute == i:
                color = "crimson"
            else:
                color = "lightBlue"
            drawCircle(posX, posY, 75, border = "black", fill = color)
            first, second = att.split(" ")
            drawLabel(f"{first}", posX, posY - 30, size = 16)
            drawLabel(f"{second}", posX, posY - 15, size = 16)
            val = rounded(100 * getattr(app, app.controllers[app.controllerSelect][att]["attr"])) / 100
            drawLabel(f"{val}", posX, posY + 20, size = 40, bold = True)

def drawNoiseBox(app):
    if app.controllerSelect != None:
        box = (-5, 100, 205, 175)
        left, top, width, height = box
        drawRect(left, top, width, height, fill = "lightGray", border = "black")
        if app.noiseSelected:
            color = "crimson"
        else:
            color = "lightBlue"
        posX = 100
        posY = 100 + 175/2
        drawCircle(posX, posY, 75, fill = color, border = "black")
        drawLabel("Sensor", posX, posY - 50, size = 20)
        drawLabel("Noise Level", posX, posY - 30, size = 20)
        drawLabel(f"{app.noiseLevel}", posX, posY + 15, size = 50)

def drawSystemParams(app):
    inc = 175
    width = 50 + len(app.labels) * inc
    drawRect(0, app.height - 200, width, 200, fill = "lightGrey", opacity = 50, border = "black")
    for i, att in enumerate(app.labels):
        posY = app.height - 100
        posX = 107.5 + i * inc
        if app.liveSelectedSystemParam == i:
            color = "crimson"
        else:
            color = "lightBlue"
        drawCircle(posX, posY, 75, border = "black", fill = color)
        first, second, third = att[0].split(" ")
        drawLabel(f"{first} {second}", posX, posY - 30, size = 16)
        drawLabel(f"{third}", posX, posY - 15, size = 16)
        val = rounded(100 * getattr(app, att[1])) / 100
        drawLabel(f"{val}", posX, posY + 20, size = 40, bold = True)

def drawCartSys(app):
    #Finding endpoint of pendulum
    endpoint = (app.posX + app.length * np.sin(app.theta), app.posY - app.length * np.cos(app.theta))

    #Drawing
    drawRect(app.posX * PPM - 50, app.posY * PPM - 25, 100, 50)
    drawLine(app.posX * PPM, app.posY * PPM, endpoint[0] * PPM, endpoint[1] * PPM)
    drawCircle(endpoint[0] * PPM, endpoint[1] * PPM, 25)
    drawCircle(app.posX * PPM - 35, app.posY * PPM + 30, 10)
    drawCircle(app.posX * PPM + 35, app.posY * PPM + 30, 10)
    drawLine(0, app.posY * PPM + 40, app.width, app.posY * PPM + 40)

def drawForces(app):
    startPoint = (app.posX * PPM, app.posY * PPM)
    perturbMag = app.perturbQ / 500
    perturbEndPoint = ((app.posX + perturbMag) * PPM, app.posY * PPM)
    drawLine(startPoint[0], startPoint[1], perturbEndPoint[0], perturbEndPoint[1], lineWidth = 5, fill = "red", arrowEnd = True)
    if app.controllerSelect != None and app.controllerEnabled:
        controlMag = app.controllerQ / 500
        controlEndPoint = ((app.posX + controlMag) * PPM, app.posY * PPM)
        drawLine(startPoint[0], startPoint[1], controlEndPoint[0], controlEndPoint[1], lineWidth = 5, fill = "green", arrowEnd = True)

###################################################################UX FUNCTIONS###################################################################
def onKeyPress(app, key):
    #Switch between start and sim screens
    if key == "s":
        app.startScreen = not app.startScreen
        resetSim(app)
    if app.startScreen:
        #Open the controls selector
        startScreenOptions(app, key)
    else:
        #Toggle the controller when in sim mode
        simScreenOptions(app, key)

def startScreenOptions(app, key):
    if key == "p":
        app.controlsPage = not app.controlsPage
    elif app.controlsPage:
        if key == "up" or key == "down":
            #Change the quantity of the selected controller attribute
            if app.paramSelected is not None and app.controllerSelect is not None:
                paramName = list(app.controllers[app.controllerSelect])[app.paramSelected]
                paramInfo = app.controllers[app.controllerSelect][paramName]
                delta = paramInfo["step"] if key == "up" else -paramInfo["step"]
                setattr(app, paramInfo["attr"], getattr(app, paramInfo["attr"]) + delta)
                if getattr(app, paramInfo["attr"]) == 0:
                    setattr(app, paramInfo["attr"], getattr(app, paramInfo["attr"]) - delta)
    else:
        #Change the quanitity of the selected physical attribute
        if key == "up" and app.selectedIndex is not None:
            attrName, step = app.labels[app.selectedIndex][1], app.labels[app.selectedIndex][2]
            setattr(app, attrName, getattr(app, attrName) + step)
        elif key == "down" and app.selectedIndex is not None:
            attrName, step = app.labels[app.selectedIndex][1], app.labels[app.selectedIndex][2]
            setattr(app, attrName, getattr(app, attrName) - step)
            if almostEqual(getattr(app, attrName), 0):
                setattr(app, attrName, getattr(app, attrName) + step)

def simScreenOptions(app, key):
    if (key == "p") and (app.controllerSelect != None):
        app.controllerEnabled = not app.controllerEnabled
        if app.controllerEnabled and app.controllerSelect == "Proportional-Integral-Derivative":
            # Reset integral windup and prime error so derivative starts at 0
            app.PIDTotal = 0
            app.PIDError = [app.measuredTheta, app.measuredVel]
        else:
            app.controllerQ = 0

    elif key == "r":
        app.posX = app.width / (2 * PPM)

    #Allow changing the controller attributes
    elif ((key == "up") or (key == "down")) and app.liveSelectedControllerAttribute != None:
        paramName = list(app.controllers[app.controllerSelect])[app.liveSelectedControllerAttribute]
        paramInfo = app.controllers[app.controllerSelect][paramName]
        delta = paramInfo["step"] if key == "up" else -paramInfo["step"]
        setattr(app, paramInfo["attr"], getattr(app, paramInfo["attr"]) + delta)
        if almostEqual(getattr(app, paramInfo["attr"]), 0):
            setattr(app, paramInfo["attr"], getattr(app, paramInfo["attr"]) - delta)
        solveForKMatrix(app)

    #Allow changing the sensor noise
    elif ((key == "up") or (key == "down")) and app.noiseSelected == True:
        delta = 1 if key == "up" else -1
        app.noiseLevel += delta
        if app.noiseLevel == -1:
            app.noiseLevel -= delta

    #Allow changing the system parameters
    elif ((key == "up") or (key == "down")) and app.liveSelectedSystemParam != None:
        attrName, step = app.labels[app.liveSelectedSystemParam][1], app.labels[app.liveSelectedSystemParam][2]
        step *= -1 if key == "up" else 1
        setattr(app, attrName, getattr(app, attrName) - step)
        if almostEqual(getattr(app, attrName), 0):
            setattr(app, attrName, getattr(app, attrName) + step)
        solveForKMatrix(app)

def onKeyHold(app, keys):
    #Perturb the cart so you can see how it responds
    if not app.startScreen:
        if "right" in keys:
            app.perturbQ = 200
        elif "left" in keys:
            app.perturbQ = -200

def onKeyRelease(app, key):
    #Allow yourself to not put force on the cart
    if not app.startScreen:
        if key == "right" or key == "left":
            app.perturbQ = 0

def onMousePress(app, mouseX, mouseY):
    if app.startScreen and app.controlsPage:
        #Selecting controller
        selectController(app, mouseX, mouseY)

        #Toggle the information screen
        toggleInfoScreen(app, mouseX, mouseY)

    elif app.startScreen:
        #Select system attribute
        selectAttribute(app, mouseX, mouseY)
    
    else:
        #Live change the controller
        selectNewController(app, mouseX, mouseY)

        #Live change controller attributes
        liveUpdateControllerAttributes(app, mouseX, mouseY)

        #Live change the sensor noise
        changeSensorNoise(app, mouseX, mouseY)

        #Live change the system params
        changeSystemParams(app, mouseX, mouseY)

def toggleInfoScreen(app, mouseX, mouseY):
    if (0 <= mouseX <= 100) and (0 <= mouseY <= 100):
        app.infoScreen = not app.infoScreen

def selectController(app, mouseX, mouseY):
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
            #Selecting attributes of the controller
            for j, paramName in enumerate(list(app.controllers[controller])):
                posY = 200 + incY * (j + 1)
                if distance(posX + 70, posY, mouseX, mouseY) <= 70:
                    app.paramSelected = j
                    paramHit = True
                    break
            #Clicking empty space detoggles
            if not paramHit:
                app.paramSelected = None
            break
    #Clicking empty space detoggles
    if not happened:
        app.paramSelected = None
        app.controllerSelect = None

def selectAttribute(app, mouseX, mouseY):
    inc = app.width / (len(app.labels) + 1)
    happened = False
    #Draws attributes
    for i in range(len(app.labels)):
        xPos = inc * (i + 1)
        if distance(xPos, app.posY * PPM, mouseX, mouseY) <= 100:
            app.selectedIndex = i
            happened = True
    if not happened:
        app.selectedIndex = None

def distance(x0, y0, x1, y1):
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

def selectNewController(app, mouseX, mouseY):
    box = (app.width - 125, 45, 100, 90)
    if box[0] <= mouseX <= box[0] + box[2]:
        if box[1] <= mouseY <= box[1] + box[3] // 3:
            app.PIDTotal = 0
            app.PIDError = [app.measuredTheta, app.measuredVel]
            app.controllerSelect = "Proportional-Integral-Derivative"
            app.workDone = 0
        elif box[1] + box[3] // 3 < mouseY <= box[1] + 2 * box[3] // 3:
            app.controllerSelect = "Linear Quadratic Regulator"
            app.workDone = 0
        elif box[1] + 2 * box[3] // 3 < mouseY <= box[1] + box[3]:
            app.ekf_state = np.array([
            app.measuredPos,
            app.measuredVel,
            app.measuredTheta,
            app.measuredThetaVel])
            app.controllerSelect = "Linear Quadratic Gaussian"
            app.workDone = 0

def liveUpdateControllerAttributes(app, mouseX, mouseY):
    if app.controllerSelect != None:
        happened = False
        inc = 175
        width = 50 + len(app.controllers[app.controllerSelect]) * inc
        leftSide = app.width - width
        for i, att in enumerate(list(app.controllers[app.controllerSelect])):
            posY = app.height - 100
            posX = leftSide + (107.5 + i * inc)
            if distance(posX, posY, mouseX, mouseY) <= 75:
                app.liveSelectedControllerAttribute = i
                happened = True
        if not happened:
            app.liveSelectedControllerAttribute = None
    
def changeSensorNoise(app, mouseX, mouseY):
    if distance(100, 100 + 175/2, mouseX, mouseY) <= 75:
        app.noiseSelected = True
    else:
        app.noiseSelected = False

def changeSystemParams(app, mouseX, mouseY):
    happened = False
    inc = 175
    width = 50 + len(app.labels) * inc
    for i in range(len(app.labels)):
        posY = app.height - 100
        posX = (107.5 + i * inc)
        if distance(posX, posY, mouseX, mouseY) <= 75:
            app.liveSelectedSystemParam = i
            happened = True
    if not happened:
        app.liveSelectedSystemParam = None

################################################################INTEGRATOR FUNCTIONS###############################################################
def onStep(app):
    if not app.startScreen:
        #Find total force
        app.Q = app.controllerQ + app.perturbQ

        #Put theta between -pi and pi
        rescaleTheta(app)

        #Verlet integration being used because total mechanical integration of the system must be preserved
        verlet(app)

        #Change from integrator parameters to measured parameters by introducing Gaussian noise
        generateGaussians(app)

        #Start running controller select
        if app.controllerSelect == "Proportional-Integral-Derivative" and app.controllerEnabled == True and not app.startScreen:
            PID(app)
        elif app.controllerSelect == "Linear Quadratic Gaussian" and app.controllerEnabled == True and not app.startScreen:
            LQG(app)
        elif app.controllerSelect == "Linear Quadratic Regulator" and app.controllerEnabled == True and not app.startScreen:
            LQR(app)

        #Update work done for display
        if app.controllerSelect != None and app.controllerEnabled:
            app.workDone += abs(app.Q) * abs(app.xDot)
    
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

################################################################CONTROLLER FUNCTIONS###############################################################
def generateGaussians(app):
    #Create the Gaussian noise distribution independenlty for each of the measurables
    noisePos = np.random.normal(app.gaussianMean, app.gaussianSTDPos * app.noiseLevel)
    noiseVel = np.random.normal(app.gaussianMean, app.gaussianSTDVel * app.noiseLevel)
    noiseTheta = np.random.normal(app.gaussianMean, app.gaussianSTDTheta * app.noiseLevel)
    noiseThetaVel = np.random.normal(app.gaussianMean, app.gaussianSTDThetaVel * app.noiseLevel)

    #Add noise to terms
    app.measuredPos = app.posX + noisePos
    app.measuredVel = app.xDot + noiseVel
    app.measuredTheta = app.theta + noiseTheta
    app.measuredThetaVel = app.thetaDot + noiseThetaVel

def PID(app):
    #Performs PID (Proportional-Integral-Derivative) control on the angle of the
    pendulumKineticEnergy = 0.5 * app.pendMass * (app.length * app.measuredThetaVel) ** 2
    pendulumPotentialEnergy = app.pendMass * app.gravity * app.length * np.cos(app.measuredTheta)
    totalPendulumEnergy = pendulumKineticEnergy + pendulumPotentialEnergy
    pendulumEnergyError = app.pendMass * app.gravity * app.length - totalPendulumEnergy

    if app.measuredTheta >= -0.65 and app.measuredTheta < 0.65 and pendulumEnergyError > -3:
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
        app.controllerQ = matrix @ state

    else:
        swingUp(app)

def swingUpForce(app):
    if app.measuredVel > 10 or app.measuredVel < -10:
        force = -400 if app.measuredVel > 0 else 400
        return force
    pendulumKineticEnergy = 0.5 * app.pendMass * (app.length * app.measuredThetaVel) ** 2
    pendulumPotentialEnergy = app.pendMass * app.gravity * app.length * np.cos(app.measuredTheta)
    totalPendulumEnergy = pendulumKineticEnergy + pendulumPotentialEnergy
    pendulumEnergyError = app.pendMass * app.gravity * app.length - totalPendulumEnergy
    k = -70 * app.pendMass/ (((app.pendMass + app.cartMass) ** 0.5) * app.length * app.gravity)
    kx = 15 / (app.pendMass + app.cartMass)
    alpha = 0.5
    kxd = 8 * (kx) ** 0.5
    force = k * app.measuredThetaVel * np.cos(app.measuredTheta) * pendulumEnergyError - kx * (1 + alpha * pendulumEnergyError) * (app.posX - app.width / (2 * PPM)) - kxd * app.measuredVel
    if pendulumEnergyError < 0:
        if force < -200:
            force = -200
        elif force > 200:
            force = 200
    return force

def swingUp(app):
    app.controllerQ = swingUpForce(app)
    #Resets integral and derivative terms for PID controller to avoid windup
    if app.controllerSelect == "Proportional-Integral-Derivative":
        app.PIDTotal = 0
        app.PIDError = [app.measuredTheta, app.measuredVel]
    #Resets Kalman filtering state to avoid windup
    elif app.controllerSelect == "Linear Quadratic Gaussian":
        app.ekf_state = np.array([
        app.measuredPos,
        app.measuredVel,
        app.measuredTheta,
        app.measuredThetaVel
    ])

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
    #Performs Kalman filter then goes right into LQR
    pendulumKineticEnergy = 0.5 * app.pendMass * (app.length * app.measuredThetaVel) ** 2
    pendulumPotentialEnergy = app.pendMass * app.gravity * app.length * np.cos(app.measuredTheta)
    totalPendulumEnergy = pendulumKineticEnergy + pendulumPotentialEnergy
    pendulumEnergyError = app.pendMass * app.gravity * app.length - totalPendulumEnergy

    if app.measuredTheta >= -0.65 and app.measuredTheta <= 0.65 and pendulumEnergyError > -3:
        app.measuredPos, app.measuredVel, app.measuredTheta, app.measuredThetaVel = kalmanFilter(app)
    LQR(app)

#This function was written by AI, but I understand the high level principals of what is going on
def kalmanFilter(app):
    dt = 1 / app.stepspersecond

    #Make dynamics based prediction of state
    f0 = kalmanDynamics(app, app.ekf_state)
    x_pred = app.ekf_state + f0 * dt

    # Numerical Jacobian of the discrete transition
    eps = 1e-5
    J = np.zeros((4, 4))
    for i in range(4):
        sp = app.ekf_state.copy()
        sp[i] += eps
        J[:, i] = (kalmanDynamics(app, sp) - f0) / eps
    F = np.eye(4) + J * dt

    #Statistics about process noise
    Q_noise = np.diag([1e-4, 1e-3, 1e-4, 1e-3])
    kalmanCovarienceMatrix = F @ app.ekf_P @ F.T + Q_noise  

    #Update trust in sensors as opposed to dynamics model
    R_noise = np.diag([(app.gaussianSTDPos * app.noiseLevel)**2, (app.gaussianSTDVel * app.noiseLevel)**2,
                       (app.gaussianSTDTheta * app.noiseLevel)**2, (app.gaussianSTDThetaVel * app.noiseLevel)**2])

    #Finds sensor measurements then computes Kalman gain
    measurements = np.array([app.measuredPos, app.measuredVel, app.measuredTheta, app.measuredThetaVel])
    K = kalmanCovarienceMatrix @ np.linalg.inv(kalmanCovarienceMatrix + R_noise)

    #Makes predictions on most accurate noise-free state is
    app.ekf_state = x_pred + K @ (measurements - x_pred)

    #Updates future noise estimate
    app.ekf_P = (np.eye(4) - K) @ kalmanCovarienceMatrix

    return tuple(app.ekf_state)

#This function was also written by AI, but I understand it fully
def kalmanDynamics(app, state):
    #Numerically solves the linearized system dynamics equations that I derived
    x, xdot, theta, thetadot = state
    
    M = np.array([
        [app.cartMass + app.pendMass,          app.pendMass * app.length * np.cos(theta)],
        [app.pendMass * app.length * np.cos(theta), app.pendMass * app.length**2]
    ])
    rhs = np.array([
        app.Q + app.pendMass * app.length * thetadot**2 * np.sin(theta),
        app.pendMass * app.gravity * app.length * np.sin(theta)
    ])
    sol = np.linalg.solve(M, rhs)  # sol = [x'', theta'']
    return np.array([xdot, sol[0], thetadot, sol[1]])

def LQR(app):
    #Performs Linear Quadratic Regulation control based on the already discovered K matrix
    pendulumKineticEnergy = 0.5 * app.pendMass * (app.length * app.measuredThetaVel) ** 2
    pendulumPotentialEnergy = app.pendMass * app.gravity * app.length * np.cos(app.measuredTheta)
    totalPendulumEnergy = pendulumKineticEnergy + pendulumPotentialEnergy
    pendulumEnergyError = app.pendMass * app.gravity * app.length - totalPendulumEnergy

    if app.measuredTheta >= -0.65 and app.measuredTheta <= 0.65 and pendulumEnergyError > -3:
        #Defines target values and their errors
        desiredPos = app.width / (2 * PPM)
        desiredVel = 0
        desiredTheta = 0
        desiredThetaVel = 0

        posError = app.measuredPos - desiredPos
        velError = app.measuredVel - desiredVel
        thetaError = app.measuredTheta - desiredTheta
        thetaVelError = app.measuredThetaVel - desiredThetaVel

        #Creates state vector then solve for control output
        state = np.array([posError, velError, thetaError, thetaVelError]).T
        app.controllerQ = -(app.KMatrix @ state)
        if app.controllerQ < -200:
            app.controllerQ = -200
        elif app.controllerQ > 200:
            app.controllerQ = 200
    else:
        swingUp(app)


runApp()