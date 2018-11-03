using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class Tanh : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return (2 / (1 + Math.Exp(-2 * x))) - 1;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return 1 - Math.Pow(Activate(x, Params), 2);
        }
    }

    public class TanhParams : ActivationParameters
    {

    }
}
