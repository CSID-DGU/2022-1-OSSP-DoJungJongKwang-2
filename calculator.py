import expressionparse

expression = 'x * x + 1'

t = expressionparse.Tree()
t.parse(expression)

print (t.toPolishNotation())
print (t.toInfixNotation())
print (t.toReversePolishNotation())

t.setVariable('x', 1)
print (t.evaluate())