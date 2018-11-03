
namespace NeuralNetworkFundamentals.Activation_Functions.Functions
{
    class SReLU : ActivationFunction
    {
        // S-shaped ReLU
        public override double Activate(double x, ActivationParameters Params)
        {
            SReLUParams temp = (SReLUParams)Params;
            if (x <= temp.TLeft)
                return temp.TLeft + temp.AlphaLeft * (x - temp.TLeft);
            if (x < temp.TRight)
                return x;
            return temp.TRight + temp.AlphaRight * (x - temp.TRight);
        }

        public override double Derivate(double x, ActivationParameters Params)
        {
            SReLUParams temp = (SReLUParams)Params;
            if (x <= temp.TLeft)
                return temp.AlphaLeft;
            if (x < temp.TRight)
                return x;
            return temp.AlphaRight;
        }
    }

    public class SReLUParams : ActivationParameters
    {
        private double tLeft;
        private double tRight;
        private double alphaLeft;
        private double alphaRight;

        public double TLeft { get => tLeft; set => tLeft = value; }
        public double TRight { get => tRight; set => tRight = value; }
        public double AlphaLeft { get => alphaLeft; set => alphaLeft = value; }
        public double AlphaRight { get => alphaRight; set => alphaRight = value; }

        public SReLUParams(double tLeft, double tRight, double alphaLeft, double alphaRight)
        {
            this.tLeft = tLeft;
            this.tRight = tRight;
            this.alphaLeft = alphaLeft;
            this.alphaRight = alphaRight;
        }
    }
}
