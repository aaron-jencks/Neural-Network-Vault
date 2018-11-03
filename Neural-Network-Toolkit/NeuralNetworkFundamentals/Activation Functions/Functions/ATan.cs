using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class ATan : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return Math.Atan(x);
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return 1 / (Math.Pow(x, 2) + 1);
        }
    }

    public class ATanParams : ActivationParameters
    {

    }
}
