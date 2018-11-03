
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class RReLU : ActivationFunction
    {
        // Randomized Leaky Rectified Linear Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            RReLUParams temp = (RReLUParams)Params;
            return (x < 0) ? temp.Alpha * x : x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            RReLUParams temp = (RReLUParams)Params;
            return (x < 0) ? temp.Alpha : 1;
        }
    }

    public class RReLUParams : ActivationParameters
    {
        private double alpha;

        public double Alpha { get => alpha; set => alpha = value; }

        public RReLUParams(double alpha)
        {
            this.alpha = alpha;
        }
    }
}
