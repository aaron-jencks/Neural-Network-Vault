using NeuralNetworkFundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkFundamentals.Windows_Form_Controls;

namespace Neural_Network_Testbench
{
    class Program
    {
        // These are the test settings, learning rate, iterations, samples, expected outputs, etc...
        static bool IsTraining = false;
        static int iterations = 2000;
        static double learningRate = 0.5;
        //static NetworkViewBox viewbox;

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

        static List<int> layerInfo = new List<int>() { 2, 2, 1 };

        static void Main(string[] args)
        {
            char keypress;
            do
            {
                Console.WriteLine("Setting up test...");
                Random rnd = new Random();

                NeuralNetwork net = new NeuralNetwork(layerInfo, learningRate: learningRate);   // Creates a network with 2 inputs, 1 hidden layer of 2, and 2 outputs
                net.TrainingUpdateEvent += OnTrainingUpdateEvent;
                net.TrainingFinishEvent += OnTrainingFinishEvent;

                // Creates a viewing box for the network.
                //viewbox = new NetworkViewBox(ref net);

                // Sets the weights and biases of the network prior to training
                // START HERE: http://web.cecs.pdx.edu/~mm/MachineLearningSpring2017/NNs.pdf On slide 42

                net.GenWeightsAndBiases();

                // Trains the network
                IsTraining = true;
                net.Train(iterations, sampleIn, sampleOut);
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
            Console.WriteLine("Sample [{0}, {1}] XOR [{2}]", result.Layers[0][0].Activation, result.Layers[0][1].Activation, result.Layers.Last()[0].Activation);
            //Console.WriteLine("Error: {0}", result.Error);
        }

        public static void OnTrainingFinishEvent(object sender)
        {
            IsTraining = false;
        }
    }
}
