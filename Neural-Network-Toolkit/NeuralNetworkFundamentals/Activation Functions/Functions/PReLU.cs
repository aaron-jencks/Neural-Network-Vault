
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class PReLU : ActivationFunction
    {
        // Parametric ReLU
        public override double Activate(double x, ActivationParameters Params)
        {
            PReLUParams temp = (PReLUParams)Params;
            return (x < 0) ? temp.Alpha * x : x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            PReLUParams temp = (PReLUParams)Params;
            return (x < 0) ? temp.Alpha : 1;
        }
    }

    public class PReLUParams : ActivationParameters
    {
        private double alpha;

        public double Alpha { get => alpha; set => alpha = value; }

        public PReLUParams(double alpha)
        {
            this.alpha = alpha;
        }
    }
}
