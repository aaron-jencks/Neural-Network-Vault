using System;

namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class SoftExponential : ActivationFunction
    {
        // Inverse Square Root Linear Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            SoftExponentialParams temp = (SoftExponentialParams)Params;
            if (temp.Alpha < 0)
                return -(Math.Log(1 - temp.Alpha * (x + temp.Alpha))) / temp.Alpha;
            if (temp.Alpha == 0)
                return x;
            return (Math.Exp(temp.Alpha * x) - 1) / temp.Alpha + temp.Alpha;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            SoftExponentialParams temp = (SoftExponentialParams)Params;
            return (temp.Alpha < 0) ? 1 / (1 - temp.Alpha * (temp.Alpha + x)) : Math.Exp(temp.Alpha * x);
        }
    }

    public class SoftExponentialParams : ActivationParameters
    {
        private double alpha;

        public double Alpha { get => alpha; set => alpha = value; }

        public SoftExponentialParams(double alpha)
        {
            this.alpha = alpha;
        }
    }
}
