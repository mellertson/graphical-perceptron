#!/usr/bin/env
from graphics import *
from abc import ABC, abstractmethod
from math import e


class ActivationFunctionBase(ABC):

	@classmethod
	@abstractmethod
	def activation_fn(cls, input_value, *args) -> float:
		"""

		:param input_value:
		:type input_value:

		:rtype: float
		"""


class SigmoidActivation(ActivationFunctionBase):

	@classmethod
	def activation_fn(cls, input_value, *args):
		return 1 / (1 + e^-input_value)


class ErrorFunctionBase(ABC):

	@classmethod
	@abstractmethod
	def error_fn(cls, target, output):
		"""
		Calculate the error of the output value with respect to the target value

		:param target:
		:type target: float | int
		:param output:
		:type output: float | int

		:rtype: float
		"""


class OutputError(ErrorFunctionBase):

	@classmethod
	def error_fn(cls, target, output):
		return (target - output) * output * (1 - output)


class HiddenError(ErrorFunctionBase):

	@classmethod
	def error_fn(cls, target, output):
		return output * (1 - output) *


class LearningFunctionBase(ABC):

	@classmethod
	@abstractmethod
	def learn_fn(cls, connection, learning_rate, error_base):
		"""
		Abstract method for a learning function used in a neural network

		:param connection: a connection between two neurons
		:type connection: Connection
		:param learning_rate:
		:type learning_rate: float
		:param error_base:
		:type error_base: ErrorFunctionBase

		:return: the updated weight of the given connection
		:rtype: Connection
		"""


class BackPropagation(LearningFunctionBase):

	@classmethod
	def learn_fn(cls, connection, learning_rate, error_base):
		"""
		Perform back-propagation on the connection at the given learning rate

		:param connection: a connection between two neurons
		:type connection: Connection
		:param learning_rate:
		:type learning_rate: float
		:param error_base:
		:type error_base: ErrorFunctionBase

		:return: the updated weight of the given connection
		:rtype: Connection
		"""
		# adjust weights of output layer neurons
		if connection.forward.n_type == 'output':
			connection.weight += learning_rate * connection.forward.error * connection.backward.output

		# TODO: adjust weights of hidden layer neurons
		elif connection.forward.n_type == 'hidden':
			pass


class Window(ABC):

	background = 'grey'
	input_color = 'blue'
	hidden_color = 'yellow'
	output_color = 'green'

	def __init__(self, title, width=500, height=500):
		self.win = GraphWin(title, width, height)
		self.set_background(self.background)

	def set_background(self, color):
		self.win.setBackground(color)

	def draw_window(self):
		"""
		Render each drawable object in the window

		:rtype: None
		"""
		for drawable in self.drawables:
			drawable.draw(self.win)

	@property
	@abstractmethod
	def drawables(self):
		"""
		Return a list of drawable objects

		:return:
		:rtype: list
		"""

	def close(self):
		self.win.getMouse()
		self.win.close()


class ModelBase(ABC):
	""" The base class for all perceptron data models """

	@property
	@abstractmethod
	def drawables(self):
		"""
		Return a list of drawable objects

		:return:
		:rtype: list
		"""


class Connection(object):
	"""
	Connects an output from a neuron to the input of a neuron

	:ivar output_value:
	:type output_value: float | int
	:ivar forward:
		The neuron the connection feeds forward to
	:type forward: Neuron
	:ivar backward:
		The neuron the connection back-propogates to
	:type backward: Neuron
	:ivar weight:
		The connections weight factor
	:type weight: float
	"""

	def __init__(self, forward, backward, weight=0.5, layer_type='hidden'):
		if layer_type == 'input':
			self.layer_type = 'input'
			self.weight = 1
		else:
			self.layer_type = layer_type
			self.weight = weight
		self.forward = forward
		self.backward = backward
		self.output_value = 1

	def predict(self, input_value, biase):
		"""

		:param input_value:
		:type input_value: float | int
		:param biase:
		:type biase: float

		:rtype: float
		"""
		self.output_value = input_value * self.weight * biase
		return self.output_value


class Neuron(object):
	"""
	A neuron in a perceptron neural-network

	:ivar n_type:
		The neuron's type, either: 'input', 'hidden<int>', 'output'
	:type n_type: str
	:ivar biase:
	:type biase: float
	:ivar activation_base:
		The neuron's activation function, for example:
		step_wise(), sigmoid(), or gaussian()
	:type activation_base: ActivationFunctionBase
	:ivar error_fn:
		The neuron's error function used during training, for example:
		mse(), or squared_error()
	:type error_fn: callable
	:ivar connections:
		A dictionary of inputs and outputs.  For example:
		{
			'inputs': [],
			'outputs': [],
		}
	:type connections: dict
	:ivar output: the last output value of the neuron
	:type output: float
	:ivar error: the error of the last output value
	:type error: float
	"""

	def __init__(self, n_type, biase, activation_base, error_fn, connections=None):
		self.n_type = n_type
		if n_type == 'input':
			self.biase = 1
		else:
			self.biase = biase
		self.activation_base = activation_base
		self.error_fn = error_fn
		if isinstance(connections, dict):
			if 'inputs' in connections and 'outputs' in connections:
				self.connections = connections
			else:
				raise ValueError("connections must have 'inputs' and 'outputs'")
		else:
			self.connections = {'inputs': [], 'outputs': []}
		self.input_value = 1
		self.output = 1
		self.error = 0

	def predict(self, input_value=None):
		if input_value:
			self.input_value = input_value
		o = 0
		for output_connection in self.connections['outputs']:
			o += output_connection.predict(input_value, self.biase)
		return self.activation_base.activation_fn(o)


class PerceptronModel(ModelBase):
	"""
	A perceptron neural network, capable of back-propogation
	"""

	def __init__(self):
		# TODO: instantiate the perceptron
		# TODO: translate the perceptron into a list of drawables
		pass

	def drawables(self):
		"""
		Return a list of drawable neurons and connections with weights and biases

		:return:
		:rtype: list
		"""
		# TODO: return the current state of this perceptron with: neurons, connections, and biases


class PerceptronApp(Window):
	"""
	Instantiate and run a perceptron model and display it in a window

	:ivar model: a data model to be rendered in the window
	:type model: ModelBase
	:ivar window: the window to display the data model in
	:type window: Window
	"""

	window_title = 'Perceptron'

	def __init__(self, model, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model

	@property
	def drawables(self):
		return self.model.drawables

	def run(self):
		"""
		Instantiate the perceptron data model and display it in a window

		:rtype: None
		"""




if __name__ == '__main__':
	app = PerceptronApp(model=PerceptronModel())
	app.run()