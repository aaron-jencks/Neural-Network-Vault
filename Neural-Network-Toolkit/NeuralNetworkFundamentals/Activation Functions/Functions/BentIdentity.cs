using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class BentIdentity : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return (Math.Sqrt(Math.Pow(x, 2) + 1) - 1) / 2 + x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return x / (2 * Math.Sqrt(Math.Pow(x, 2) + 1)) + 1;
        }
    }

    public class BentIdentityParams : ActivationParameters
    {

    }
}
