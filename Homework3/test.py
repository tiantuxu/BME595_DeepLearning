from neural_network import NeuralNetwork

from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

And = AND()
And.train()
print "======== Test cases for AND ========"
print "AND(False, False) = ", And(False, False)
print "AND(False, True) = ", And(False, True)
print "AND(True, False) = ", And(True, False)
print "AND(True, True) = ", And(True, True)
print ""

Or = OR()
Or.train()
print "========== Test cases for OR ========"
print "OR(False, False) = ", Or(False, False)
print "OR(False, True) = ", Or(False, True)
print "OR(True, False) = ", Or(True, False)
print "OR(True, True) = ", Or(True, True)
print ""

Not = NOT()
Not.train()
print "======== Test cases for NOT ========"
print "NOT(False) = ", Not(False)
print "NOT(True) = ", Not(True)
print ""

Xor = XOR()
Xor.train()
print "======== Test cases for XOR ========"
print "XOR(False, False) = ", Xor(False, False)
print "XOR(False, True) = ", Xor(False, True)
print "XOR(True, False) = ", Xor(True, False)
print "XOR(True, True) = ", Xor(True, True)
