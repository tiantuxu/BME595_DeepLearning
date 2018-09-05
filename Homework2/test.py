from neural_network import NeuralNetwork

from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

AND = AND()
OR = OR()
NOT = NOT()
XOR = XOR()

print "======== Test cases for AND ========"
print "And(False, False) = ", AND(False, False)
print "And(False, True) = ", AND(False, True)
print "And(True, False) = ", AND(True, False)
print "And(True, True) = ", AND(True, True)
print ""

print "========== Test cases for OR ========"
print "Or(False, False) = ", OR(False, False)
print "Or(False, True) = ", OR(False, True)
print "Or(True, False) = ", OR(True, False)
print "Or(True, True) = ", OR(True, True)
print ""

print "======== Test cases for NOT ========"
print "NOT(False) = ", NOT(False)
print "NOT(True) = ", NOT(True)
print ""

print "======== Test cases for XOR ========"
print "XOR(False, False) = ", XOR(False, False)
print "XOR(False, True) = ", XOR(False, True)
print "XOR(True, False) = ", XOR(True, False)
print "XOR(True, True) = ", XOR(True, True)
