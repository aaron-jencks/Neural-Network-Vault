
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class Identity : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            return x;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            return 1;
        }
    }

    public class IdentityParams : ActivationParameters
    {

    }
}
