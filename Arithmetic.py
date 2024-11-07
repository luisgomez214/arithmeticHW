'''
This is a "simple" homework to practice parsing grammars and working with the resulting parse tree.
'''


import lark
from lark import Lark, Token, Tree, visitors


grammar = r"""
    start: sum

    ?sum: product
        | sum "+" product  -> add
        | sum "-" product  -> sub
        | product "%" atom -> mod
        | sum product      -> mul  // Implicit multiplication

    ?product: power
        | product "*" power -> mul
        | product "/" power -> div

    ?power: atom
        | power "**" atom -> pow

    ?atom: NUMBER          -> number
        | "(" sum ")"    -> paren
        | "-" atom       -> unary
        | atom "(" sum ")"        -> implicit_mul 

    NUMBER: /-?[0-9]+/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""



parser = lark.Lark(grammar)


class Interpreter(lark.visitors.Interpreter):
   
    def start(self, tree):
        return self.visit(tree.children[0])

    def number(self, tree):
        return int(tree.children[0].value)

    def unary(self, tree):
        return -self.visit(tree.children[0])

    def add(self, tree):
        return self.visit(tree.children[0]) + self.visit(tree.children[1])

    def sub(self, tree):
        return self.visit(tree.children[0]) - self.visit(tree.children[1])

    def mul(self, tree):
        return self.visit(tree.children[0]) * self.visit(tree.children[1])

    def div(self, tree):
        return self.visit(tree.children[0]) // self.visit(tree.children[1])

    def paren(self, tree):
        return self.visit(tree.children[0])

    def mod(self, tree):
        return self.visit(tree.children[0]) % self.visit(tree.children[1])
    
    def pow(self, tree):
        base = self.visit(tree.children[0])
        exponent = self.visit(tree.children[1])
        
        if exponent < 0:
            return 0
        return base ** exponent

    def implicit_mul(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 * v1


    '''
    Compute the value of the expression.
    The interpreter class processes nodes "top down",
    starting at the root and recursively evaluating subtrees.

    FIXME:
    Get all the test cases to pass.


    >>> interpreter = Interpreter()
    >>> interpreter.visit(parser.parse("1"))
    1
    >>> interpreter.visit(parser.parse("-1"))
    -1
    >>> interpreter.visit(parser.parse("1+2"))
    3
    >>> interpreter.visit(parser.parse("1-2"))
    -1
    >>> interpreter.visit(parser.parse("(1+2)*3"))
    9
    >>> interpreter.visit(parser.parse("1+2*3"))
    7
    >>> interpreter.visit(parser.parse("1*2+3"))
    5
    >>> interpreter.visit(parser.parse("1*(2+3)"))
    5
    >>> interpreter.visit(parser.parse("(1*2)+3*4*(5-6)"))
    -10
    >>> interpreter.visit(parser.parse("((1*2)+3*4)*(5-6)"))
    -14
    >>> interpreter.visit(parser.parse("(1*(2+3)*4)*(5-6)"))
    -20
    >>> interpreter.visit(parser.parse("((1*2+(3)*4))*(5-6)"))
    -14

    NOTE:
    The grammar for the arithmetic above should all be implemented correctly.
    The arithmetic expressions below, however, will require you to modify the grammar.
    


    Modular division:

    >>> interpreter.visit(parser.parse("1%2"))
    1
    >>> interpreter.visit(parser.parse("3%2"))
    1
    >>> interpreter.visit(parser.parse("(1+2)%3"))
    0

    Exponentiation:

    >>> interpreter.visit(parser.parse("2**1"))
    2
    >>> interpreter.visit(parser.parse("2**2"))
    4
    >>> interpreter.visit(parser.parse("2**3"))
    8
    >>> interpreter.visit(parser.parse("1+2**3"))
    9
    >>> interpreter.visit(parser.parse("(1+2)**3"))
    27
    >>> interpreter.visit(parser.parse("1+2**3+4"))
    13
    >>> interpreter.visit(parser.parse("(1+2)**(3+4)"))
    2187
    >>> interpreter.visit(parser.parse("(1+2)**3-4"))
    23

    NOTE:
    The calculator is designed to only work on integers.
    Division uses integer division,
    and exponentiation should use integer exponentiation when the exponent is negative.
    (That is, it should round the fraction down to zero.)

    >>> interpreter.visit(parser.parse("2**-1"))
    0
    >>> interpreter.visit(parser.parse("2**(-1)"))
    0
    >>> interpreter.visit(parser.parse("(1+2)**(3-4)"))
    0
    >>> interpreter.visit(parser.parse("1+2**(3-4)"))
    1
    >>> interpreter.visit(parser.parse("1+2**(-3)*4"))
    1

    Implicit multiplication:

    >>> interpreter.visit(parser.parse("1+2(3)"))
    7
    >>> interpreter.visit(parser.parse("1(2(3))"))
    6
    >>> interpreter.visit(parser.parse("(1)(2)(3)"))
    6
    >>> interpreter.visit(parser.parse("(1)(2)+(3)"))
    5
    >>> interpreter.visit(parser.parse("(1+2)(3+4)"))
    21
    >>> interpreter.visit(parser.parse("(1+2)(3(4))"))
    36
    '''


class Simplifier(lark.Transformer):

    def start(self, children):
        return children[0]    

    def unary(self, children):
        return -children[0]

    def number(self,children):
        return int(children[0].value) 
    
    def add(self, children):
        return children[0] + children[1]

    def sub(self, children):
        return children[0] - children[1]

    def mul(self, children):
        return children[0] * children[1]

    def div(self, children):
        return children[0] // children[1]

    def paren(self, children):
        return children[0]

    def mod(self, children):
        return children[0] % children[1]

    def pow(self, children):
        base = children[0]
        exponent = children[1]

        if exponent < 0:
            return 0
        return base ** exponent

    def implicit_mul(self, children):
        return children[0] * children[1]

    '''
    Compute the value of the expression.
    The lark.Transformer class processes nodes "bottom up",
    starting at the leaves and ending at the root.
    In general, the Transformer class is less powerful than the Interpreter class.
    But in the case of simple arithmetic expressions,
    both classes can be used to evaluate the expression.

    FIXME:
    This class contains all of the same test cases as the Interpreter class.
    You should fix all the failing test cases.
    You shouldn't need to make any additional modifications to the grammar beyond what was needed for the interpreter class.
    You should notice that the functions in the lark.Transformer class are simpler to implement because you do not have to manage the recursion yourself.

    >>> simplifier = Simplifier()
    >>> simplifier.transform(parser.parse("1"))
    1
    >>> simplifier.transform(parser.parse("-1"))
    -1
    >>> simplifier.transform(parser.parse("1+2"))
    3
    >>> simplifier.transform(parser.parse("1-2"))
    -1
    >>> simplifier.transform(parser.parse("(1+2)*3"))
    9
    >>> simplifier.transform(parser.parse("1+2*3"))
    7
    >>> simplifier.transform(parser.parse("1*2+3"))
    5
    >>> simplifier.transform(parser.parse("1*(2+3)"))
    5
    >>> simplifier.transform(parser.parse("(1*2)+3*4*(5-6)"))
    -10
    >>> simplifier.transform(parser.parse("((1*2)+3*4)*(5-6)"))
    -14
    >>> simplifier.transform(parser.parse("(1*(2+3)*4)*(5-6)"))
    -20
    >>> simplifier.transform(parser.parse("((1*2+(3)*4))*(5-6)"))
    -14

    Modular division:

    >>> simplifier.transform(parser.parse("1%2"))
    1
    >>> simplifier.transform(parser.parse("3%2"))
    1
    >>> simplifier.transform(parser.parse("(1+2)%3"))
    0

    Exponentiation:

    >>> simplifier.transform(parser.parse("2**1"))
    2
    >>> simplifier.transform(parser.parse("2**2"))
    4
    >>> simplifier.transform(parser.parse("2**3"))
    8
    >>> simplifier.transform(parser.parse("1+2**3"))
    9
    >>> simplifier.transform(parser.parse("(1+2)**3"))
    27
    >>> simplifier.transform(parser.parse("1+2**3+4"))
    13
    >>> simplifier.transform(parser.parse("(1+2)**(3+4)"))
    2187
    >>> simplifier.transform(parser.parse("(1+2)**3-4"))
    23

    Exponentiation with negative exponents:

    >>> simplifier.transform(parser.parse("2**-1"))
    0
    >>> simplifier.transform(parser.parse("2**(-1)"))
    0
    >>> simplifier.transform(parser.parse("(1+2)**(3-4)"))
    0
    >>> simplifier.transform(parser.parse("1+2**(3-4)"))
    1
    >>> simplifier.transform(parser.parse("1+2**(-3)*4"))
    1

    Implicit multiplication:

    >>> simplifier.transform(parser.parse("1+2(3)"))
    7
    >>> simplifier.transform(parser.parse("1(2(3))"))
    6
    >>> simplifier.transform(parser.parse("(1)(2)(3)"))
    6
    >>> simplifier.transform(parser.parse("(1)(2)+(3)"))
    5
    >>> simplifier.transform(parser.parse("(1+2)(3+4)"))
    21
    >>> simplifier.transform(parser.parse("(1+2)(3(4))"))
    36
    '''


########################################
# other transformations
########################################


def minify(expr):

    class RemoveParentheses(lark.Transformer):
        def paren(self, children):
            return children[0]  # Remove unnecessary parentheses

        def start(self, children):
            return children[0]

    class ToString(lark.Transformer):
        def start(self, children):
            return ''.join(children)

        def number(self, children):
            return children[0].value
        
        def add(self, children):
            return f"{children[0]}+{children[1]}"

        def sub(self, children):
            return f"{children[0]}-{children[1]}"
        
        def mul(self, children):
            return f"{children[0]}*{children[1]}"
        
        def div(self, children):
            return f"{children[0]}/{children[1]}"
        
        def mod(self, children):
            return f"{children[0]}%{children[1]}"
        
        def pow(self, children):
            return f"{children[0]}**{children[1]}"

    parsed = parser.parse(expr)
    no_parens = RemoveParentheses().transform(parsed)
    return ToString().transform(no_parens)

    '''
    "Minifying" code is the process of removing unnecessary characters.
    In our arithmetic language, this means removing unnecessary whitespace and unnecessary parentheses.
    It is common to minify code in order to save disk space and bandwidth.
    For example, google penalizes a web site's search ranking if they don't minify their html/javascript code.

    FIXME:
    Implement this function so that the test cases below pass.

    HINT:
    My solution uses two lark.Transformer classes.
    The first one takes an AST and removes any unneeded parentheses.
    The second taks an AST and converts the AST into a string.
    You can solve this problem by calling parser.parse,
    and then applying the two transformers above to the resulting AST.

    NOTE:
    It is important that these types of "syntactic" transformations use the Transformer class and not the Interpreter class.
    If we used the Interpreter class, we could "accidentally do too much computation",
    but the Transformer class's leaf-to-root workflow prevents this class of bug.

    NOTE:
    The test cases below do not require any of the "new" features that you are required to add to the Arithmetic grammar.
    It only uses the features in the starting code.

    >>> minify("1 + 2")
    '1+2'
    >>> minify("1 + ((((2))))")
    '1+2'
    >>> minify("1 + (2*3)")
    '1+2*3'
    >>> minify("1 + (2/3)")
    '1+2/3'
    >>> minify("(1 + 2)*3")
    '(1+2)*3'
    >>> minify("(1 - 2)*3")
    '(1-2)*3'
    >>> minify("(1 - 2)+3")
    '1-2+3'
    >>> minify("(1 + 2)+(3 + 4)")
    '1+2+3+4'
    >>> minify("(1 + 2)*(3 + 4)")
    '(1+2)*(3+4)'
    >>> minify("1 + (((2)*(3)) + 4)")
    '1+2*3+4'
    >>> minify("1 + (((2)*(3)) + 4 * ((5 + 6) - 7))")
    '1+2*3+4*(5+6-7)'
    '''


def infix_to_rpn(expr):
    '''
    This function takes an expression in standard infix notation and converts it into an expression in reverse polish notation.
    This type of translation task is commonly done by first converting the input expression into an AST (i.e. by calling parser.parse),
    and then simplifying the AST in a leaf-to-root manner (i.e. using the Transformer class).

    HINT:
    If you need help understanding reverse polish notation,
    see the eval_rpn function.

    >>> infix_to_rpn('1')
    '1'
    >>> infix_to_rpn('1+2')
    '1 2 +'
    >>> infix_to_rpn('1-2')
    '1 2 -'
    >>> infix_to_rpn('(1+2)*3')
    '1 2 + 3 *'
    >>> infix_to_rpn('1+2*3')
    '1 2 3 * +'
    >>> infix_to_rpn('1*2+3')
    '1 2 * 3 +'
    >>> infix_to_rpn('1*(2+3)')
    '1 2 3 + *'
    >>> infix_to_rpn('(1*2)+3+4*(5-6)')
    '1 2 * 3 + 4 5 6 - * +'
    '''
        
    # Operator precedence and associativity
    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2,
        '%': 2,
        '**': 3  # Exponentiation has the highest precedence
    }
    
    # Right associativity for exponentiation
    right_associative = {'**'}

    def greater_precedence(op1, op2):
        if op1 in right_associative:
            return precedence[op1] > precedence[op2]
        return precedence[op1] >= precedence[op2]
    
    # Output queue (RPN) and operator stack
    output = []
    operators = []
    
    # Tokenize the input expression (split by spaces or detect operators)
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isdigit() or (expr[i] == '-' and (i == 0 or expr[i-1] in '([')):
            num = ''
            while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                num += expr[i]
                i += 1
            tokens.append(num)
        elif expr[i] in '+-*/%()**':
            if expr[i:i+2] == '**':  # Handling the exponentiation operator '**'
                tokens.append('**')
                i += 2
            else:
                tokens.append(expr[i])
                i += 1
        else:
            i += 1  # Skip spaces
    
    for token in tokens:
        if token.isdigit():  # If it's a number, add it to output
            output.append(token)
        elif token == '(':  # Left parenthesis, push it onto the stack
            operators.append(token)
        elif token == ')':  # Right parenthesis, pop from stack until '('
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # Pop the '('
        else:  # Operator
            while (operators and operators[-1] != '(' and 
                   (greater_precedence(operators[-1], token) or 
                    precedence.get(token, 0) == precedence.get(operators[-1], 0))):
                output.append(operators.pop())
            operators.append(token)
    
    # Pop any remaining operators from the stack
    while operators:
        output.append(operators.pop())

    # Join the output as a string with spaces between the elements
    return ' '.join(output)


def eval_rpn(expr):
    '''
    This function evaluates an expression written in RPN.

    RPN (Reverse Polish Notation) is an alternative syntax for arithmetic.
    It was widely used in the first scientific calculators because it is much easier to parse than standard infix notation.
    For example, parentheses are never needed to disambiguate order of operations.
    Parsing of RPN is so easy, that it is usually done at the same time as evaluation without a separate parsing phase.
    More complicated languages (like the infix language above) are basically always implemented with separate parsing/evaluation phases.

    You can find more details on wikipedia: <https://en.wikipedia.org/wiki/Reverse_Polish_notation>.

    NOTE:
    There is nothing to implement for this function,
    it is only provided as a reference for understanding the infix_to_rpn function.

    >>> eval_rpn("1")
    1
    >>> eval_rpn("1 2 +")
    3
    >>> eval_rpn("1 2 -")
    1
    >>> eval_rpn("1 2 + 3 *")
    9
    >>> eval_rpn("1 2 3 * +")
    7
    >>> eval_rpn("1 2 * 3 +")
    5
    >>> eval_rpn("1 2 3 + *")
    5
    >>> eval_rpn("1 2 * 3 + 4 5 6 - * +")
    9
    '''
    tokens = expr.split()
    stack = []
    operators = {
        '+': lambda a, b: a+b,
        '-': lambda a, b: a-b,
        '*': lambda a, b: a*b,
        '/': lambda a, b: a//b,
        }
    for token in tokens:
        if token not in operators.keys():
            stack.append(int(token))
        else:
            assert len(stack) >= 2
            v1 = stack.pop()
            v2 = stack.pop()
            stack.append(operators[token](v1, v2))
    assert len(stack) == 1
    return stack[0]






if __name__ == "__main__":
    interpreter = Interpreter()  # Create an instance of Interpreter
    simplifier = Simplifier()   
 

    # Testing the Interpreter
    print(interpreter.visit(parser.parse("1")))  # Should output: 1
    print(interpreter.visit(parser.parse("-1")))  # Should output: -1
    print(interpreter.visit(parser.parse("1+2")))  # Should output: 3
    print(interpreter.visit(parser.parse("1-2")))  # Should output: -1
    print(interpreter.visit(parser.parse("(1+2)*3")))  # Should output: 9
    print(interpreter.visit(parser.parse("1+2*3")))  # Should output: 7
    print(interpreter.visit(parser.parse("1*2+3")))  # Should output: 5
    print(interpreter.visit(parser.parse("1*(2+3)")))  # Should output: 5
    print(interpreter.visit(parser.parse("(1*2)+3*4*(5-6)")))  # Should output: -10
    print(interpreter.visit(parser.parse("((1*2)+3*4)*(5-6)")))  # Should output: -14
    print(interpreter.visit(parser.parse("(1*(2+3)*4)*(5-6)")))  # Should output: -20
    print(interpreter.visit(parser.parse("((1*2+(3)*4))*(5-6)")))  # Should output: -14

    # Modular division tests for Interpreter
    print(interpreter.visit(parser.parse("1%2")))  # Should output: 1
    print(interpreter.visit(parser.parse("3%2")))  # Should output: 1
    print(interpreter.visit(parser.parse("(1+2)%3")))  # Should output: 0

    # Exponentiation tests for Interpreter
    print(interpreter.visit(parser.parse("2**1")))  # Should output: 2
    print(interpreter.visit(parser.parse("2**2")))  # Should output: 4
    print(interpreter.visit(parser.parse("2**3")))  # Should output: 8
    print(interpreter.visit(parser.parse("1+2**3")))  # Should output: 9
    print(interpreter.visit(parser.parse("(1+2)**3")))  # Should output: 27
    print(interpreter.visit(parser.parse("1+2**3+4")))  # Should output: 13
    print(interpreter.visit(parser.parse("(1+2)**(3+4)")))  # Should output: 2187
    print(interpreter.visit(parser.parse("(1+2)**3-4")))  # Should output: 23

    # Exponentiation with negative exponents for Interpreter
    print(interpreter.visit(parser.parse("2**-1")))  # Should output: 0
    print(interpreter.visit(parser.parse("2**(-1)")))  # Should output: 0
    print(interpreter.visit(parser.parse("(1+2)**(3-4)")))  # Should output: 0
    print(interpreter.visit(parser.parse("1+2**(3-4)")))  # Should output: 1
    print(interpreter.visit(parser.parse("1+2**(-3)*4")))  # Should output: 1

    # Implicit multiplication tests for Interpreter
    print(interpreter.visit(parser.parse("1+2(3)")))  # Should output: 7
    print(interpreter.visit(parser.parse("1(2(3))")))  # Should output: 6
    print(interpreter.visit(parser.parse("(1)(2)(3)")))  # Should output: 6
    print(interpreter.visit(parser.parse("(1)(2)+(3)")))  # Should output: 5
    print(interpreter.visit(parser.parse("(1+2)(3+4)")))  # Should output: 21
    print(interpreter.visit(parser.parse("(1+2)(3(4))")))  # Should output: 36


    print(simplifier.transform(parser.parse("1")))  # Should output: 1
    print(simplifier.transform(parser.parse("-1")))  # Should output: -1
    print(simplifier.transform(parser.parse("1+2")))  # Should output: 3
    print(simplifier.transform(parser.parse("1-2")))  # Should output: -1
    print(simplifier.transform(parser.parse("(1+2)*3")))  # Should output: 9
    print(simplifier.transform(parser.parse("1+2*3")))  # Should output: 7
    print(simplifier.transform(parser.parse("1*2+3")))  # Should output: 5
    print(simplifier.transform(parser.parse("1*(2+3)")))  # Should output: 5
    print(simplifier.transform(parser.parse("(1*2)+3*4*(5-6)")))  # Should output: -10
    print(simplifier.transform(parser.parse("((1*2)+3*4)*(5-6)")))  # Should output: -14
    print(simplifier.transform(parser.parse("(1*(2+3)*4)*(5-6)")))  # Should output: -20
    print(simplifier.transform(parser.parse("((1*2+(3)*4))*(5-6)")))  # Should output: -14

    # Modular division tests
    print(simplifier.transform(parser.parse("1%2")))  # Should output: 1
    print(simplifier.transform(parser.parse("3%2")))  # Should output: 1
    print(simplifier.transform(parser.parse("(1+2)%3")))  # Should output: 0

    # Exponentiation tests
    print(simplifier.transform(parser.parse("2**1")))  # Should output: 2
    print(simplifier.transform(parser.parse("2**2")))  # Should output: 4
    print(simplifier.transform(parser.parse("2**3")))  # Should output: 8
    print(simplifier.transform(parser.parse("1+2**3")))  # Should output: 9
    print(simplifier.transform(parser.parse("(1+2)**3")))  # Should output: 27
    print(simplifier.transform(parser.parse("1+2**3+4")))  # Should output: 13
    print(simplifier.transform(parser.parse("(1+2)**(3+4)")))  # Should output: 2187
    print(simplifier.transform(parser.parse("(1+2)**3-4")))  # Should output: 23

    # Exponentiation with negative exponents
    print(simplifier.transform(parser.parse("2**-1")))  # Should output: 0
    print(simplifier.transform(parser.parse("2**(-1)")))  # Should output: 0
    print(simplifier.transform(parser.parse("(1+2)**(3-4)")))  # Should output: 0
    print(simplifier.transform(parser.parse("1+2**(3-4)")))  # Should output: 1
    print(simplifier.transform(parser.parse("1+2**(-3)*4")))  # Should output: 1

    # Implicit multiplication tests
    print(simplifier.transform(parser.parse("1+2(3)")))  # Should output: 7
    print(simplifier.transform(parser.parse("1(2(3))")))  # Should output: 6
    print(simplifier.transform(parser.parse("(1)(2)(3)")))  # Should output: 6
    print(simplifier.transform(parser.parse("(1)(2)+(3)")))  # Should output: 5
    print(simplifier.transform(parser.parse("(1+2)(3+4)")))  # Should output: 21
    print(simplifier.transform(parser.parse("(1+2)(3(4))")))



    print(minify("1 + 2"))
    print(minify("1 + ((((2))))"))
    print(minify("1 + (2*3)"))
    print(minify("1 + (2/3)"))
    print(minify("(1 + 2)*3"))
    print(minify("(1 - 2)*3"))
    print(minify("(1 - 2)+3"))
    print(minify("(1 + 2)+(3 + 4)"))
    print(minify("(1 + 2)*(3 + 4)"))
    print(minify("1 + (((2)*(3)) + 4)"))
    print(minify("1 + (((2)*(3)) + 4 * ((5 + 6) - 7))"))


    print(infix_to_rpn('1'))
    print(infix_to_rpn('1+2'))
    print(infix_to_rpn('1-2'))
    print( infix_to_rpn('(1+2)*3'))
    print(infix_to_rpn('1+2*3'))
    print(infix_to_rpn('1*2+3'))
    print(infix_to_rpn('1*(2+3)'))
    print(infix_to_rpn('(1*2)+3+4*(5-6)'))


    print(eval_rpn("1"))
    print(eval_rpn("1 2 +"))
    print(eval_rpn("1 2 -"))
    print(eval_rpn("1 2 + 3 *")) 
    print(eval_rpn("1 2 3 * +"))
    print(eval_rpn("1 2 * 3 +"))
    print(eval_rpn("1 2 3 + *"))
    print(eval_rpn("1 2 * 3 + 4 5 6 - * +"))


