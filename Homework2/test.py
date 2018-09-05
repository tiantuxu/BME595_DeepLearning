from neural_network import NeuralNetwork

from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

And = AND()
Or = OR()
Not = NOT()
Xor = XOR()

print "======== Test cases for AND ========"
print "And(False, False) = ", And(False, False)
print "And(False, True) = ", And(False, True)
print "And(True, False) = ", And(True, False)
print "And(True, True) = ", And(True, True)
print ""

print "========== Test cases for OR ========"
print "Or(False, False) = ", Or(False, False)
print "Or(False, True) = ", Or(False, True)
print "Or(True, False) = ", Or(True, False)
print "Or(True, True) = ", Or(True, True)
print ""

print "======== Test cases for NOT ========"
print "NOT(False) = ", Not(False)
print "NOT(True) = ", Not(True)
print ""

print "======== Test cases for XOR ========"
print "XOR(False, False) = ", Xor(False, False)
print "XOR(False, True) = ", Xor(False, True)
print "XOR(True, False) = ", Xor(True, False)
print "XOR(True, True) = ", Xor(True, True)
