namespace NeuralNetworkFundamentals.Activation_Functions
{
    public class ActivationParameters { };
    public abstract class ActivationFunction
    {
        // For information on how to figure out the standard activation function algorithms
        // Go here: https://en.wikipedia.org/wiki/Activation_function

        public abstract double Activate(double x, ActivationParameters Params);

        public abstract double Derivate(double x, ActivationParameters Params);
    }
}
