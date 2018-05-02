
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class ReLU : ActivationFunction
    {
        // Rectified Linear Unit
        public override double Activate(double x, ActivationParameters Params)
        {
            return (x < 0) ? 0 : x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return (x < 0) ? 0 : 1;
        }
    }

    public class ReLUParams : ActivationParameters
    {
    }
}
