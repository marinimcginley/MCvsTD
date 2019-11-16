import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# For Monte Carlo
QStar = np.zeros((10, 21, 2))
# For TD Sarsa
Q = np.zeros((10, 21, 2))
E = np.zeros((10, 21, 2))

llambda = 0

# Learning Curve for epsilon 0 and 1
LC1 = []
LC2 = []

gameOver = False

# N[s,a]
NStateAction = np.zeros((10, 21, 2))
# N[s]
NState = np.zeros((10, 21))

def dealerPlays(sum):
	while True:
		card, color = generateCard()
		if (color == 1 and sum - card < 1 or color > 1 and sum + card > 21):
			return "bust"

		if (color == 1):
			sum -= card
		else:
			sum += card

		if (sum >= 17):
			return sum

# Returns s' and r
def step(s,a):
	# If action is "hit"
	if (a == 1):
		card, color = generateCard()

		if (color == 1):
			newS = s[0], s[1] - card
		else:
			newS = s[0], s[1] + card

		if (checkIfCardIsSuitable(card, color, s)):
			return newS, 0
		else:
			return s, -1
	# If action is "stick"
	else:
		#sticking to the sum
		result = dealerPlays(s[0])
		global gameOver
		gameOver = True
		if (result == "bust"):
			return s, 1
		else:
			if (result > s[1]):
				return s, -1
			elif (result == s[1]):
				return s, 0
			else:
				return s, 1

# Returns true if new card does not lead to a bust
# Else returns false
def checkIfCardIsSuitable(card, color, s):
	if (color == 1 and s[1] - card < 1 or color > 1 and s[1] + card > 21):
		return False
	return True

#color; 1: red, 2-3: black
def generateCard():
	card = random.randint(1, 10)
	color = random.randint(1, 3)
	return card, color

# returns alpha for a given state and action
def calculateAlpha(s, a):
	return 1 / NStateAction[s[0]-1, s[1]-1, a]

# returns alpha for a given state
def calculateEpsilonGreedy(s):  
	return 100 / (100 + NState[s[0]-1, s[1]-1])

# returns s'
def monteCarlo(s):
	NState[s[0]-1, s[1]-1] += 1

	a = findAction(QStar, s)

	NStateAction[s[0]-1, s[1]-1, a] += 1

	newS, R = step(s, a)	
	updateQStar(s, a, R)
	return newS

# Updates our policy
def updateQStar(s, a, R):
	alpha = calculateAlpha(s, a)
	QStar[s[0]-1, s[1]-1, a] = QStar[s[0]-1, s[1]-1, a] + alpha * (R - QStar[s[0]-1, s[1]-1, a])

# Returns action
# 0 for "hit"
# 1 for "stick"
def findAction(matrix, s):
    epsilon = calculateEpsilonGreedy(s)
    if (matrix[s[0]-1, s[1]-1, 0] == 0 and matrix[s[0]-1, s[1]-1, 1] == 0):
    	# Random if state is unknown
    	return random.randint(0,1)
    elif (np.random.uniform() < epsilon):
    	# Exploration
    	if (matrix[s[0]-1, s[1]-1, 0] > matrix[s[0]-1, s[1]-1, 1]):
    		return 1
    	else:
    		return 0
    else:
    	# Exploitation
    	if (matrix[s[0]-1, s[1]-1, 0] > matrix[s[0]-1, s[1]-1, 1]):
    		return 0
    	else:
    		return 1

def playMC(numberOfTimesPlayed):
	for x in range(numberOfTimesPlayed):
		s = [random.randint(1, 10), random.randint(1, 10)]
		global gameOver
		while (gameOver == False):
			newS = monteCarlo(s)
			s = newS		
		gameOver = False

# Returns s' and a'
def sarsa(s, a):
	NState[s[0]-1, s[1]-1] += 1
	NStateAction[s[0]-1, s[1]-1, a] += 1
	newS, R = step(s, a)
	global Q, E, llambda
	newA = findAction(Q, newS)

	alpha = calculateAlpha(s, a)
	delta = calculateDelta(R, newS, newA, s, a)

	E[s[0]-1, s[1]-1, a] += 1

	Q = np.add(Q, alpha * delta * E)
	E = llambda * E

	return newS, newA

def calculateDelta(R, newS, newA, s, a):
	return R + Q[newS[0]-1, newS[1]-1, newA] - Q[s[0]-1, s[1]-1, a]

def playTD(numberOfTimesPlayed):
	for x in range(numberOfTimesPlayed):
		s = [random.randint(1, 10), random.randint(1, 10)]
		E = np.zeros((10, 21, 2))
		global gameOver, Q, LC1, LC2, llambda
		a = findAction(Q, s)
		while (gameOver == False):

			newS, newA = sarsa(s, a)
			s = newS
			a = newA
		
		gameOver = False

		if (llambda == 0):
			LC1.append(calculateMeanSquaredError()/(21*10*2))
		if (llambda == 1):
			LC2.append(calculateMeanSquaredError()/(21*10*2))

# returns a matrix for the value functon
def calculateValueFunction(matrix):
	return np.amax(matrix, axis = 2)

def calculateMeanSquaredError():
	# what is going on here???
	return np.sum(np.square(np.subtract(Q, QStar)))

def plotMeanSquaredError(MES):
	y, x = np.split(MES, 2, axis = 1)

	fig, ax = plt.subplots()
	ax.plot(x, y)

	ax.set(xlabel='Lambda', ylabel='Mean-Squared Error', title='Mean-squared Error plotted against lambda')
	ax.grid()
	
	fig.savefig('Mean-squared_Error_plotted_against_lambda.png')

def plotMeanSquaredErrorAgainstEpisode():
	fig, ax = plt.subplots()
	ax.plot(np.arange(1, 1001), LC1, label="lambda = 0")
	ax.plot(np.arange(1, 1001), LC2, label="lambda = 1")

	ax.set_xlabel('Episodes')
	ax.set_ylabel('Mean-Squared Error')
	ax.set_title('Mean-squared Error plotted against episodes')
	ax.legend(loc=2)
	ax.grid()
	
	fig.savefig('Learning_Curve_of_Mean-Squared-Error_against_Episodes.png')

def drawMCValue(VStar, Qstring):
	x = range(10)
	y = range(21)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	X, Y = np.meshgrid(x, y)
	ax.plot_surface(X, Y, VStar.T)
	
	plt.xlabel('First card of dealer', fontsize = 10)
	plt.ylabel('Sum of player', fontsize = 10)
	stringg = str(Qstring) + ' learning for Easy 21'
	plt.title(stringg)
	stringa = 'ValueFunction_for_Episodes_100000' + str(Qstring) + '.png'
	plt.savefig(stringa)


def play():
	playMC(100000)
	VStar = calculateValueFunction(QStar)
	drawMCValue(VStar, 'Monte Carlo')	
	
	MES = np.zeros((11, 2))
	MES1 = np.zeros((11, 2))

	# á að vera range(11)
	for x in range(11):
		NState = np.zeros((10, 21))
		NStateAction = np.zeros((10, 21, 2))
		global Q, llambda
		Q = np.zeros((10, 21, 2))

		playTD(1000)

		MES[x, 0] = calculateMeanSquaredError()/(21*10*2)
		MES[x, 1] = llambda

		llambda += 0.1
		llambda = round(llambda, 1)

	plotMeanSquaredError(MES)
	plotMeanSquaredErrorAgainstEpisode()
	
play()