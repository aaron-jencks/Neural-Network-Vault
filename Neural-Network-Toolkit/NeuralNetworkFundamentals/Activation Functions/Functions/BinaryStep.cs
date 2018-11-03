
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class BinaryStep : ActivationFunction
    {
        public override double Activate(double x, ActivationParameters Params)
        {
            double temp = 0;
            if (x >= 0)
                temp = 1;
            return temp;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            // the real return value for x==0 is ?, but you can't code that.
            return (x == 0) ? x : 0;
        }
    }

    public class BinaryStepParams : ActivationParameters
    {

    }
}
