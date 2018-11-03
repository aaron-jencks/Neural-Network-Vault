using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class ISRLU : ActivationFunction
    {
        // Inverse Square Root Linear Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            ISRLUParams temp = (ISRLUParams)Params;
            return (x < 0) ? x / Math.Sqrt(1 + temp.Alpha * Math.Pow(x, 2)) : x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            ISRLUParams temp = (ISRLUParams)Params;
            return (x < 0) ? Math.Pow(1 / Math.Sqrt(1 + temp.Alpha * Math.Pow(x, 2)), 3) : 1;
        }
    }

    public class ISRLUParams : ActivationParameters
    {
        private double alpha;

        public double Alpha { get => alpha; set => alpha = value; }

        public ISRLUParams(double alpha)
        {
            this.alpha = alpha;
        }
    }
}
