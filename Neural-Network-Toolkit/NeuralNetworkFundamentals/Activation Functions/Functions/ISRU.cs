using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class ISRU : ActivationFunction
    {
        // Inverse Square Root Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            ISRUParams temp = (ISRUParams)Params;
            return x / Math.Sqrt(1 + temp.Alpha * Math.Pow(x, 2));
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            ISRUParams temp = (ISRUParams)Params;
            return Math.Pow(1 / Math.Sqrt(1 + temp.Alpha * Math.Pow(x, 2)), 3);
        }
    }

    public class ISRUParams : ActivationParameters
    {
        private double alpha;

        public double Alpha { get => alpha; set => alpha = value; }

        public ISRUParams(double alpha)
        {
            this.alpha = alpha;
        }
    }
}
