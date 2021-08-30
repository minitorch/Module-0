import minitorch.operators as operators


class MathTest:
    @staticmethod
    def neg(a):
        return -a

    @staticmethod
    def addConstant(a):
        return 5 + a

    @staticmethod
    def subConstant(a):
        return a - 5

    @staticmethod
    def mult(a):
        return 5 * a

    @staticmethod
    def div(a):
        return a / 5

    @staticmethod
    def inv(a):
        return operators.inv(a + 3.5)

    @staticmethod
    def sig(a):
        return operators.sigmoid(a)

    @staticmethod
    def log(a):
        return operators.log(a + 100000)

    @staticmethod
    def relu(a):
        return operators.relu(a + 5.5)

    @staticmethod
    def exp(a):
        return operators.exp(a - 200)

    @staticmethod
    def add2(a, b):
        return a + b

    @staticmethod
    def mul2(a, b):
        return a * b

    @staticmethod
    def div2(a, b):
        return a / (b + 5.5)

    @staticmethod
    def gt2(a, b):
        return operators.lt(b, a + 1.2)

    @staticmethod
    def lt2(a, b):
        return operators.lt(a + 1.2, b)

    @staticmethod
    def eq2(a, b):
        return operators.eq(a, (b + 5.5))

    @staticmethod
    def sum_red(a):
        return operators.sum(a)

    @staticmethod
    def mean_red(a):
        return operators.sum(a) / float(len(a))

    @staticmethod
    def mean_full_red(a):
        return operators.sum(a) / float(len(a))

    @classmethod
    def _tests(cls):
        """
        Returns a list of all the math tests.
        """
        one_arg = []
        two_arg = []
        red_arg = []
        for k in dir(MathTest):
            if callable(getattr(MathTest, k)) and not k.startswith("_"):
                base_fn = getattr(MathTest, k)
                scalar_fn = getattr(cls, k)
                tup = (k, base_fn, scalar_fn)
                if k.endswith("2"):
                    two_arg.append(tup)
                elif k.endswith("red"):
                    red_arg.append(tup)
                else:
                    one_arg.append(tup)
        return one_arg, two_arg, red_arg


class MathTestVariable(MathTest):
    @staticmethod
    def inv(a):
        return 1.0 / (a + 3.5)

    @staticmethod
    def sig(x):
        return x.sigmoid()

    @staticmethod
    def log(x):
        return (x + 100000).log()

    @staticmethod
    def relu(x):
        return (x + 5.5).relu()

    @staticmethod
    def exp(a):
        return (a - 200).exp()

    @staticmethod
    def sum_red(a):
        return a.sum(0)

    @staticmethod
    def mean_red(a):
        return a.mean(0)

    @staticmethod
    def mean_full_red(a):
        return a.mean()

    @staticmethod
    def eq2(a, b):
        return a == (b + 5.5)

    @staticmethod
    def gt2(a, b):
        return a + 1.2 > b

    @staticmethod
    def lt2(a, b):
        return a + 1.2 < b
