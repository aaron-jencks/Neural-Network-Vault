using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class ELU : ActivationFunction
    {
        // Exponential Linear Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            ELUParams temp = (ELUParams)Params;
            return (x < 0) ? temp.Alpha * (Math.Exp(x) - 1) : x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            ELUParams temp = (ELUParams)Params;
            return (x < 0) ? Activate(x, Params) + temp.Alpha : 1;
        }
    }

    public class ELUParams : ActivationParameters
    {
        private double alpha;

        public double Alpha { get => alpha; set => alpha = value; }

        public ELUParams(double alpha)
        {
            this.alpha = alpha;
        }
    }
}
