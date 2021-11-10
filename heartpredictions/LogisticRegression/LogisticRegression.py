from torch import nn

"""
LogisticRegression Class

LogisticRegression class inheritated from torch.nn.Module.

"""

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, final_activation=None):
        """
        Parameters:
            input_dim (int) : Input dimension.
            output_dim (int) : Output dimension.
            final_activation (layer) : The final activation function.

        Atributes:
            linear (layer): The linear layer.
            final_activation (layer): The activation function.
        """

        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.final_activation = final_activation

    def forward(self, x):
        """
        Forward of the pytorch model (build the model).

        Parameters:
            x (Tensor) : The input.

        Returns:
            out (Tensor)
        """
        out = self.linear(x)

        if self.final_activation != None:
            out = self.final_activation(out)

        return out
