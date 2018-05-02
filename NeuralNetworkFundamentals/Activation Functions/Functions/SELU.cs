using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class SELU : ActivationFunction
    {
        // Scaled Exponential Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            SELUParams temp = (SELUParams)Params;
            return temp.Lambda * ((x < 0) ? temp.Alpha * (Math.Exp(x) - 1) : x);
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            SELUParams temp = (SELUParams)Params;
            return temp.Lambda * ((x < 0) ? temp.Alpha * Math.Exp(x) : 1);
        }
    }

    public class SELUParams : ActivationParameters
    {
        private double alpha;
        private double lambda;

        public double Alpha { get => alpha; set => alpha = value; }
        public double Lambda { get => lambda; set => lambda = value; }

        public SELUParams(double alpha = 1.67326, double lambda = 1.0507)
        {
            this.alpha = alpha;
            this.lambda = lambda;
        }
    }
}
