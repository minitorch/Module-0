"""
"""


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we all it to hold
    any value for testing.
    """

    def __init__(self, x=None):
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def __repr__(self):
        return repr(self.value)


class Module:
    """Modules are a recursive tree-shaped data-structure. Each module can
    store three things: 1) parameters, 2) non-parameter data, 3) other
    modules. Internally the user stores each of these directly on `self`,
    but the module spies under the hood to determine the type of each
    assignment.

    Here is an example of the simplest usage of a module::

      class MyModule(Module):
          def __init__(self, arg):
              # Initialize the super-class (so it can spy.)
              super().__init__()

                # A parameter member (subclass of Parameter)
              self.parameter1 = Parameter(15)

              # Another member
              self.data = 25

              # A module member (subclass of Module)
              self.sub_module = OtherModule(arg, arg+10)

    Warning:
        All subclasses must begin their initialization by calling ::

           super().__init__()

        This allows the module to capture any members of type :class:`Module` or :class:`Parameter`
        and store them in a special dictionary.


    Internally, parameters (type 1) are stored in :attr:`_parameters`, data (type 2)
    is stored on `self`, modules (type 3) are stored in :attr:`_modules`.


    The main benefit of this infrastructure is that it allows us to `flatten` a module
    to get out all of its parameters using :func:`named_parameters`.  Here is an example of
    how you can create a tree of modules and then extract the flattened parameters::

      class Module1(Module):
          def __init__(self, arg):
              super().__init__()
              self.p1 = Parameter(5)
              self.a = Module2()

      class Module2(Module):
          def __init__(self, arg):
              super().__init__()
              self.p2 = Parameter(10)

      class Module3(Module):
          def __init__(self, arg):
              super().__init__()
              self.b = Module4()

      class Module4(Module):
          def __init__(self, arg):
              super().__init__()
              self.p3 = Parameter(15)

      np = Module1().named_parameters()
      assert np["b.c.p3"].value == 15

    .. image:: figs/Module/module.png

    Additionally a module can have a :attr:`mode` indicating how it is
    currently being used. The mode should propagate to all of its
    children. For simplicity, we only consider a train and eval mode.


    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        mode (string): Mode of operation, can be {"train", "eval"}.

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.mode = "train"

    def modules(self):
        "Return the child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement.
        raise NotImplementedError

    def eval(self):
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement.
        raise NotImplementedError

    def named_parameters(self):
        """
        Collect all the ancestor parameters of this module.


        Returns:
            dict: Each name (key) and :class:`Parameter` (value) under this module.
        """
        # TODO: Implement.
        raise NotImplementedError

    def parameters(self):
        return self.named_parameters().values()

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        return self.__getattribute__(key)

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
