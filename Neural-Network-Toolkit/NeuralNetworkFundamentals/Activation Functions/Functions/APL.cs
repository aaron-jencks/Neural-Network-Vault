
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class APL : ActivationFunction
    {
        // Adaptive Piecewise Linear
        public override double Activate(double x, ActivationParameters Params)
        {
            APLParams temp = (APLParams)Params;
            double total = (x < 0) ? 0 : x;
            for (int i = 1; i <= temp.S; i++)
                total += temp.Alpha * ((-x + temp.Beta < 0) ? 0 : -x + temp.Beta);
            return total;
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            APLParams temp = (APLParams)Params;
            double total = 0;
            for (int i = 1; i <= temp.S; i++)
                total += temp.Alpha * HeavySideStep(-x + temp.Beta);
            return HeavySideStep(x) - total;
        }

        private double HeavySideStep(double x)
        {
            if (x < 0)
                return 0;
            if (x == 0)
                return 0.5;
            return 1;
        }
    }

    public class APLParams : ActivationParameters
    {
        private int s;
        private double alpha;
        private double beta;

        public int S { get => s; set => s = value; }
        public double Alpha { get => alpha; set => alpha = value; }
        public double Beta { get => beta; set => beta = value; }

        public APLParams(int s, double alpha, double beta)
        {
            this.s = s;
            this.alpha = alpha;
            this.beta = beta;
        }
    }
}
