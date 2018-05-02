using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkFundamentals;
using NeuralNetworkFundamentals.Windows_Form_Controls;
using System.Windows.Forms;

namespace Neural_Network_Testbench_W_form
{
    class Program
    {
        // These are the test settings, learning rate, iterations, samples, expected outputs, etc...
        static bool IsTraining = false;
        static int iterations = 1000;
        static int delay = 3;
        static double learningRate = 0.5;

        static List<List<double>> sampleIn = new List<List<double>>()
        {
            new List<double>(){0, 0 },
            new List<double>(){0, 1 },
            new List<double>(){1, 0 },
            new List<double>(){1, 1 }
        };

        static List<List<double>> sampleOut = new List<List<double>>()
        {
            new List<double>(){0},
            new List<double>(){1},
            new List<double>(){1},
            new List<double>(){0}
        };

        static List<int> layerInfo = new List<int>() { 2, 3, 1 };

        [STAThread]
        static void Main(string[] args)
        {
            char keypress;

            NeuralNetwork net = new NeuralNetwork(layerInfo, learningRate: learningRate);   // Creates a network with 2 inputs, 1 hidden layer of 2, and 2 outputs
            net.TrainingUpdateEvent += OnTrainingUpdateEvent;
            net.TrainingFinishEvent += OnTrainingFinishEvent;

            Task.Factory.StartNew(() =>
            {
                Application.EnableVisualStyles();
                Application.Run(new NetworkViewBox(ref net, 10));
            });

            do
            {
                Console.WriteLine("Setting up test...");
                Random rnd = new Random();

                // Sets the weights and biases of the network prior to training
                // START HERE: http://web.cecs.pdx.edu/~mm/MachineLearningSpring2017/NNs.pdf On slide 42

                net.GenWeightsAndBiases();

                // Trains the network
                IsTraining = true;
                net.Train(iterations, sampleIn, sampleOut, delay: delay);
                while (IsTraining) ;

                //Console.Clear();
                Console.WriteLine("Training Complete!");
                Console.WriteLine("Testing inputs");
                foreach (List<double> sample in sampleIn)
                {
                    List<double> temp = net.Calc(sample);
                    Console.WriteLine("For Inputs [{0}, {1}]:", sample[0], sample[1]);
                    //Console.WriteLine("The output is: {0}, {1}\n", temp[0], temp[1]);
                    Console.WriteLine("The output is: {0}\n", temp[0]);
                }

                keypress = Console.ReadKey().KeyChar;

            } while (keypress != 's');
        }

        public static void OnTrainingUpdateEvent(object sender, TrainingUpdateEventArgs result)
        {
            // Executes every time the network finishes a training sample
            //Console.WriteLine("Finished Iteration " + result.Iteration);
            Console.Title = "Iteration: " + result.Iteration;
            Console.WriteLine("Sample [{0}, {1}] XOR [{2}]", result.Layers[0][0].Activation, result.Layers[0][1].Activation, result.Layers.Last()[0].Activation);
            //Console.WriteLine("Error: {0}", result.Error);
        }

        public static void OnTrainingFinishEvent(object sender)
        {
            IsTraining = false;
        }
    }
}
