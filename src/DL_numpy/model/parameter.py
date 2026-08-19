import numpy as np


class Parameter:
    """
    Trainable Parameter

    Stores:
        data : Trainable weights
        grad : Gradient of loss w.r.t. weights (None until backward is called)
    """

    def __init__(self, data):
        # Trainable values forced to float64 for stability
        self.data = np.asarray(data, dtype=np.float64)

        # Gradient is None until backward() accumulates into it
        self.grad = None

    # -----------------------------
    # Reset Gradients
    # -----------------------------
    def zero_grad(self):
        """
        Reset gradient to zero if initialized.
        """
        if self.grad is not None:
            self.grad.fill(0.0)

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def numel(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    # -----------------------------
    # String Representation
    # -----------------------------
    def __str__(self):
        grad_norm = np.linalg.norm(self.grad) if self.grad is not None else 0.0

        return (
            f"Parameter(\n"
            f"  shape      = {self.shape}\n"
            f"  numel      = {self.numel}\n"
            f"  dtype      = {self.dtype}\n"
            f"  grad_norm  = {grad_norm:.6f}\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()