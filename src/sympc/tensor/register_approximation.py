"""Register decorator that keeps track of all the approximations we have."""


class RegisterApproximation:
    """Used to keep track of all the approximations we have.

    Arguments:
        nfunc: the name of the function.

    This class is used as a register decorator class that keeps track of all the
    approximation functions we have in a dictionary.

    """

    approx_dict = {}

    def __init__(self, nfunc):
        """Initializer for the RegisterApproximation class.

        Arguments:
            nfunc: the name of the function.
        """
        self.nfunc = nfunc

    def __call__(self, func):
        """Returns a wrapper functions that adds an approximation function to the dictionary approx_dict.

        Arguments:
            func: function to be added to the dictionary approx_dict.

        Returns:
            wrapper functions of function func that was added to the approx_dict dictionary.

        """
        self.approx_dict[self.nfunc] = func

        def wrapper(*args, **kwargs):

            res = func(*args, **kwargs)
            return res

        return wrapper
