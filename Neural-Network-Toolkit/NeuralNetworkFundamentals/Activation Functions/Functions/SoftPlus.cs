using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class Softplus : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return 1 / (1 + Math.Exp(-x));
        }
    }

    public class SoftplusParams : ActivationParameters
    {

    }
}
