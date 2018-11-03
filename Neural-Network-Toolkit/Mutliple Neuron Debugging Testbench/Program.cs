using System;
using System.Collections.Generic;
using NeuralNetworkFundamentals;
using static NeuralNetworkFundamentals.Neuron;


namespace Single_Neuron_Debugging_Testbench
{
    class Program
    {
        static bool activationFlag;
        static long outputID;

        static void Main(string[] args)
        {
            Random rnd = new Random();
            int numInputs = 2;
            double learningRate = 0.5;
            List<double> weightInit1 = new List<double>() { rnd.NextDouble(), rnd.NextDouble() };
            List<double> weightInit3 = new List<double>() { rnd.NextDouble(), rnd.NextDouble() };
            List<double> weightInit2 = new List<double>() { rnd.NextDouble(), rnd.NextDouble() };
            // Creates the input to the test neuron
            List<Neuron> inputs = new List<Neuron>(numInputs);
            for (int i = 0; i < inputs.Capacity; i++)
            {
                inputs.Add(new Neuron());
            }
            // Creates an output layer neuron
            Neuron neuronTest1 = new Neuron(inputs,
                weightInit1);
            Neuron neuronTest3 = new Neuron(inputs,
                weightInit3);
            Neuron neuronTest2 = new Neuron(new List<Neuron>() { neuronTest1, neuronTest3 },
                weightInit2,
                outputLayer: true);
            //Neuron neuronTest = new Neuron(inputs, outputLayer: true);
            //neuronTest.RandomizeWeights();

            outputID = neuronTest2.ID;
            neuronTest2.ActiveEvent += OnActivate;
            neuronTest1.ActiveEvent += OnActivate;

            // Our testing sample values
            List<List<double>> input = new List<List<double>>() {
                new List<double>() { 0, 0 },
                new List<double>() { 0, 1 },
                new List<double>() { 1, 0 },
                new List<double>() { 1, 1 } };
            List<double> expectedOutput = new List<double>() { 0, 1, 1, 0 };
            char key;

            Console.WriteLine("For this test, the first inputs is: {0} and {1}", input[0][0], input[0][1]);
            Console.WriteLine("The second inputs is: {0} and {1}", input[1][0], input[1][1]);
            Console.WriteLine("The third inputs is: {0} and {1}", input[2][0], input[2][1]);
            Console.WriteLine("The fourth inputs is: {0} and {1}", input[3][0], input[3][1]);
            Console.WriteLine("The expected outputs will be: {0} and {1} and {2} and {3}", expectedOutput[0], expectedOutput[1], expectedOutput[2], expectedOutput[3]);
            Console.WriteLine("The initial weights are {0}:{2}, {3}:{4}, and {1}", neuronTest1.Weights[0], neuronTest2.Weights[0], neuronTest1.Weights[1],
                neuronTest3.Weights[0], neuronTest3.Weights[1]);
            Console.WriteLine("The initial bias is {0}, {2}, and {1}\n", neuronTest1.Bias, neuronTest2.Bias, neuronTest3.Bias);
            Console.WriteLine("The learning rate is {0}", learningRate);
            Console.WriteLine("The momentum is 1");

            long iteration = 0;
            do
            {
                int currentSet = 0;
                while (currentSet++ <= 10000)
                {
                    Console.WriteLine("Iteration {0}", iteration++);
                    // Prepare the test
                    for (int epoch = 0; epoch < input.Count; epoch++)
                    {
                        //Console.WriteLine("For this sample, the input is: {0} and {1}", input[epoch][0], input[epoch][1]);
                        activationFlag = true;
                        for (int i = 0; i < inputs.Count; i++)
                        {
                            // Activate each neuron in the input layer.
                            inputs[i].RawInput = input[epoch][i];
                            inputs[i].Activate();
                        }
                        while (activationFlag) ;    // Wait for the activation to finish.

                        // Checks and reports the error of the neuron to the console
                        neuronTest2.AssignDelta(1, learningRate, expectedOutput[epoch], AdjustValues: false);    // Calculates the derivatives of the neuron's error
                        double prevBias2 = neuronTest2.Bias;
                        double prevWeight2 = neuronTest2.Weights[0];
                        neuronTest2.AdjustValues(1, learningRate, expectedOutput[epoch]);                        // Assigns the gradient to the weight and bias.

                        neuronTest1.AssignDelta(1, learningRate, 0, new List<Neuron>() { neuronTest2 }, false);    // Calculates the derivatives of the neuron's error
                        double prevBias1 = neuronTest1.Bias;
                        double prevWeight1 = neuronTest1.Weights[0];
                        neuronTest1.AdjustValues(1, learningRate, expectedOutput[epoch]);                        // Assigns the gradient to the weight and bias.

                        neuronTest3.AssignDelta(1, learningRate, 0, new List<Neuron>() { neuronTest2 }, false);    // Calculates the derivatives of the neuron's error
                        double prevBias3 = neuronTest3.Bias;
                        double prevWeight3 = neuronTest3.Weights[0];
                        neuronTest3.AdjustValues(1, learningRate, expectedOutput[epoch]);                        // Assigns the gradient to the weight and bias.

                        if (currentSet - 1 == 10000)
                        {
                            Console.WriteLine("For this sample, the input is: {0} and {1}", input[epoch][0], input[epoch][1]);
                            Console.WriteLine("The output of the  output neuron is {0} and should be {1}", neuronTest2.Activation, expectedOutput[epoch]);
                            
                            Console.WriteLine("The output of the first hidden neuron is {0} and should be {1}", neuronTest1.Activation, neuronTest2.Weights[0] * neuronTest2.Delta);
                            
                            Console.WriteLine("The output of the second hidden neuron is {0} and should be {1}", neuronTest3.Activation, neuronTest2.Weights[1] * neuronTest2.Delta);
                            
                        }
                    }
                }

                key = Console.ReadKey().KeyChar;

            } while (key != 's');
        }

        static void OnActivate(object sender, ActivationEventArgs e)
        {
            Neuron temp = (Neuron)sender;
            if(temp.ID == outputID)
                activationFlag = false;
        }
    }
}
