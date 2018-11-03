using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class Softsign : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return x / (1 + Math.Abs(x));
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return 1 / Math.Pow(1 + Math.Abs(x), 2);
        }
    }

    public class SoftsignParams : ActivationParameters
    {

    }
}
