namespace NeuralNetworkFundamentals.Activation_Functions
{
    /// <summary>
    /// Class for implementing parameters to be passed into activation functions, if need be.
    /// </summary>
    public class ActivationParameters { };

    /// <summary>
    /// Class for implementing activation functions from
    /// </summary>
    public abstract class ActivationFunction
    {
        // For information on how to figure out the standard activation function algorithms
        // Go here: https://en.wikipedia.org/wiki/Activation_function

        /// <summary>
        /// Activates the function
        /// </summary>
        /// <param name="x">input value to use</param>
        /// <param name="Params">parameters to use</param>
        /// <returns></returns>
        public abstract double Activate(double x, ActivationParameters Params);

        /// <summary>
        /// Returns the derivative of the function for a given value
        /// </summary>
        /// <param name="x"></param>
        /// <param name="Params"></param>
        /// <returns></returns>
        public abstract double Derivate(double x, ActivationParameters Params);
    }
}
