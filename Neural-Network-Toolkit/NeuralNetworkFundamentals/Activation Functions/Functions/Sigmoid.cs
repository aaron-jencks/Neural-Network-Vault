using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class Sigmoid : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return Activate(x, Params) * (1 - Activate(x, Params));
        }
    }

    public class SigmoidParams : ActivationParameters
    {

    }
}
