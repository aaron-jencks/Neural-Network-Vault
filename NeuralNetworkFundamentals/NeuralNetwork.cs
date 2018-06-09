using NeuralNetworkFundamentals.Activation_Functions;
using NeuralNetworkFundamentals.Activation_Functions.Functions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using Troschuetz.Random;
using System.Xml;
using System.Xml.Linq;
using System.Xml.XPath;

namespace NeuralNetworkFundamentals
{
    /// <summary>
    /// Training update event arguments
    /// </summary>
    public class TrainingUpdateEventArgs : EventArgs
    {
        #region properties
        private int iteration;
        private int sampleNum;
        private List<List<Neuron>> layers;
        private double error;
        private bool finished;
        #endregion

        #region Accessor Methods
        /// <summary>
        /// Iteration of the training execution
        /// </summary>
        public int Iteration { get => iteration; set => iteration = value; }

        /// <summary>
        /// Sample number of the current iteration
        /// </summary>
        public int SampleNum { get => sampleNum; set => sampleNum = value; }

        /// <summary>
        /// Total error of the current sample
        /// </summary>
        public double Error { get => error; set => error = value; }

        /// <summary>
        /// The layers of neurons for the current sample
        /// </summary>
        public List<List<Neuron>> Layers { get => layers; set => layers = value; }

        /// <summary>
        /// Flags if the training has just completed
        /// </summary>
        public bool Finished { get => finished; set => finished = value; }
        #endregion

        public TrainingUpdateEventArgs(int iteration, int sampleNum, List<List<Neuron>> layers, double error, bool finished)
        {
            this.iteration = iteration;
            this.sampleNum = sampleNum;
            this.layers = layers;
            this.error = error;
            this.finished = finished;
        }
    }

    public class NeuralNetwork
    {
        #region Static Methods
        /// <summary>
        /// Clones the neural network
        /// </summary>
        /// <param name="net">Neural net to clone</param>
        /// <returns>A new instance of the same neural network</returns>
        public static NeuralNetwork Clone(NeuralNetwork net)
        {
            // Creates a copy of the passed in neural network and returns it.
            NeuralNetwork temp = new NeuralNetwork();
            temp = net;
            return temp;
        }
        #endregion

        #region Event Information

        /// <summary>
        /// Delegate for the training update event
        /// </summary>
        /// <param name="sender">neural network sending the update</param>
        /// <param name="e">training update event args</param>
        public delegate void TrainingUpdateEventHandler(object sender, TrainingUpdateEventArgs e);

        /// <summary>
        /// Sent every time that the network finishes a sample while training the network
        /// </summary>
        public event TrainingUpdateEventHandler TrainingUpdateEvent; // Triggered every time this network finishes a sample during training.

        /// <summary>
        /// Method for triggering the training update event
        /// </summary>
        /// <param name="e">arguments to send</param>
        public void OnTrainingUpdateEvent(TrainingUpdateEventArgs e)
        {
            TrainingUpdateEvent?.Invoke(this, e);
        }

        /// <summary>
        /// Delegate for the training completion event
        /// </summary>
        /// <param name="sender">the neural network that just finished its training</param>
        public delegate void TrainingFinishEventHandler(object sender);

        /// <summary>
        /// Triggered when the neural network finishes its training execution
        /// </summary>
        public event TrainingFinishEventHandler TrainingFinishEvent; // Triggered every time this network finishes training.

        /// <summary>
        /// Method for triggering the training completion event
        /// </summary>
        public void OnTrainingFinishEvent()
        {
            TrainingFinishEvent?.Invoke(this);
        }

        #endregion

        #region Properties
        private List<List<Neuron>> layers;      // The collection of physical layers of the neural network

        /// <summary>
        /// Number of neurons that have activated, used for training
        /// </summary>
        private int activationCount;

        /// <summary>
        /// Flags if the network has not subscribed to the neurons' activation events yet
        /// </summary>
        private bool hasSubscribed = false; // state of whether the network has subscribed to the neurons' activation events or not.

        private double learningRate;
        private double momentum;

        /// <summary>
        /// Private thread used for training the network asynchronously
        /// </summary>
        private Thread trainingThread;

        private long id;
        private static long netCount = 0;
        #endregion

        #region Constructor
        /// <summary>
        /// Constructor for the neural network
        /// </summary>
        /// <param name="LayerInfo">List of integers representing the neuron count for each layer beginning with the input and ending with the output</param>
        /// <param name="defaultActivationFunction">List of default activation functions for each layer (List of sigmoid)</param>
        /// <param name="Params">List of corresponding activation parameters for each layer (List of sigmoid)</param>
        /// <param name="learningRate">learning rate for the network (0.5)</param>
        /// <param name="momentum">momentum for the network (0)</param>
        public NeuralNetwork(List<int> LayerInfo, List<ActivationFunction> defaultActivationFunction = null, List<ActivationParameters> Params = null,
            double learningRate = 0.5, double momentum = 0)
        {
            // Creates a neural network with LayerInfo.Count layers and each Layer with int neurons.
            id = netCount++;

            this.learningRate = learningRate;
            this.momentum = momentum;

            layers = new List<List<Neuron>>(LayerInfo.Count);

            if(defaultActivationFunction == null)
            {
                ////Console.WriteLine("Created the default activation functions");
                defaultActivationFunction = new List<ActivationFunction>(LayerInfo.Count);
                for (int i = 0; i < LayerInfo.Count; i++)
                    defaultActivationFunction.Add(new Sigmoid());
            }

            if(Params == null)
            {
                Params = new List<ActivationParameters>(LayerInfo.Count);
                for (int i = 0; i < LayerInfo.Count; i++)
                    Params.Add(new SigmoidParams());
            }

            // Generates the layers of Neurons
            for(int i = 0; i < LayerInfo.Count; i++)
            {
                List<Neuron> temp = new List<Neuron>(LayerInfo[i]);
                if (i == 0)
                    for (int j = 0; j < LayerInfo[i]; j++)
                        temp.Add(new Neuron(defaultActivation: defaultActivationFunction[i], defaultParameters: Params[i]));        // Creates the input layer
                else
                {
                    List<Neuron> prev = layers[i - 1];
                    for (int j = 0; j < LayerInfo[i]; j++)
                        temp.Add(new Neuron(prev,
                            defaultActivation: defaultActivationFunction[i],
                            defaultParameters: Params[i],
                            outputLayer: (i == LayerInfo.Count - 1)));  // Generates the rest of the layers
                }
                layers.Add(temp);
            }
        }

        /// <summary>
        /// Constructor for the neural network ONLY TO BE USED BY WINDOWS FORMS
        /// </summary>
        public NeuralNetwork()
        {

        }
        #endregion

        #region Accessor Methods
        /// <summary>
        /// The 2D map of all of the neurons in the network
        /// </summary>
        public List<List<Neuron>> Layers { get => layers; set => layers = value; }

        /// <summary>
        /// learning rate of the network
        /// </summary>
        public double LearningRate { get => learningRate; set => learningRate = value; }

        /// <summary>
        /// current activation levels of all of the output neurons
        /// </summary>
        public List<double> Output { get => getOutputs(); }

        /// <summary>
        /// Used to get the list of activation levels for the output neurons
        /// </summary>
        /// <returns>Returns the list of activation levels for the output neurons</returns>
        private List<double> getOutputs()
        {
            List<double> temp = new List<double>(Layers.Last().Count);
            foreach (Neuron neuron in layers.Last())
                temp.Add(neuron.Activation);
            return temp;
        }

        /// <summary>
        /// Gets the output value for a given input value
        /// </summary>
        /// <param name="inputs">input values to use</param>
        /// <returns>returns the calculated output values</returns>
        public List<double> Calc(List<double> inputs)
        {
            // Runs the network through its forward cycle and returns the outputs
            LoadSample(inputs);
            ForwardPropagate();
            List<double> temp = new List<double>(layers.Last().Count);
            foreach(Neuron neuron in layers.Last())
            {
                temp.Add(neuron.Activation);
            }

            return temp;
        }

        /// <summary>
        /// Generates, or sets, the weight and bias matrix of the network
        /// </summary>
        /// <param name="weights">Weight matrix to use (randomized)</param>
        /// <param name="biases">Bias matrix to use (randomized)</param>
        public void GenWeightsAndBiases(List<List<List<double>>> weights = null, List<List<double>> biases = null)
        {
            // Can allow the controller to generate the biases and weights prior to training.
            
            try
            {
                
                // Sets up the Normal Distribution random number generator
                NormalDistribution rndNorm = new NormalDistribution();
                rndNorm.Sigma = 0.5;
                rndNorm.Mu = 0;

                // Sets up the binomial distribution random number generator
                BinomialDistribution rndBin = new BinomialDistribution();

                // Assigns the biases, and weights
                for (int i = 0; i < layers.Count; i++)
                {
                    for (int j = 0; j < layers[i].Count; j++)
                    {
                        // Initializes the network's biases and weights
                        if (weights == null)
                            layers[i][j].RandomizeWeights(rndNorm);
                        else
                            layers[i][j].Weights = weights[i][j];

                        if (biases == null)
                            layers[i][j].RandomizeBias(rndBin);
                        else
                            layers[i][j].Bias = biases[i][j];
                    }
                }
            }
            catch(Exception e)
            {
            
                Random rnd = new Random();
                foreach(List<Neuron> layer in layers)
                    foreach(Neuron neuron in layer)
                    {
                        neuron.RandomizeBias(rnd);
                        neuron.RandomizeWeights(rnd);
                    }
            }
        }

        /// <summary>
        /// Weight matrix for the network
        /// </summary>
        public List<List<List<double>>> Weights { get => GetWeights(); set => GenWeights(value); }

        /// <summary>
        /// Bias matrix for the network
        /// </summary>
        public List<List<double>> Biases { get => GetBiases(); set => GenBiases(value); }

        /// <summary>
        /// Uniqe ID assigned to this network at initialization
        /// </summary>
        public long ID { get => id; set => id = value; }

        /// <summary>
        /// Current count of networks in existence
        /// </summary>
        public static long NetCount { get => netCount; set => netCount = value; }

        /// <summary>
        /// Momentum of the network
        /// </summary>
        public double Momentum { get => momentum; set => momentum = value; }

        /// <summary>
        /// Number of neurons in this network
        /// </summary>
        public int NeuronCount { get => GetNeuralCount(); }
        #endregion

        #region Methods

        /// <summary>
        /// Generates a weight matrix
        /// </summary>
        /// <param name="weights">Weight matrix to use (randomized)</param>
        protected virtual void GenWeights(List<List<List<double>>> weights = null)
        {
            // Can allow the controller to generate the biases and weights prior to training.
            /*
            try
            {
                // Sets up the Normal Distribution random number generator
                NormalDistribution rndNorm = new NormalDistribution();
                rndNorm.Sigma = 0.05;
                rndNorm.Mu = 0;

                // Assigns the biases, and weights
                for (int i = 1; i < layers.Count - 1; i++)
                {
                    for (int j = 0; j < layers[i].Count; j++)
                    {
                        // Initializes the network's biases and weights

                        if (weights == null)
                            layers[i][j].RandomizeWeights(rndNorm);
                        else
                            layers[i][j].Weights = weights[i][j];
                    }
                }
            }
            catch (Exception e)
            {
            */
                // Troschuetz.Random isn't working, use Random instead.
                foreach (List<Neuron> layer in layers)
                    foreach (Neuron neuron in layer)
                        neuron.RandomizeWeights(new Random());
            //}
        }

        /// <summary>
        /// Generates a bias matrix
        /// </summary>
        /// <param name="biases">Bias matrix to use (randomized)</param>
        protected virtual void GenBiases(List<List<double>> biases = null)
        {
            // Can allow the controller to generate the biases and weights prior to training.
            /*
            try
            {
                // Sets up the binomial distribution random number generator
                BinomialDistribution rndBin = new BinomialDistribution();

                // Assigns the biases, and weights
                for (int i = 0; i < layers.Count; i++)
                {
                    for (int j = 0; j < layers[i].Count; j++)
                    {
                        // Initializes the network's biases and weights
                        if (biases == null)
                            layers[i][j].RandomizeBias(rndBin);
                        else
                            layers[i][j].Bias = biases[i][j];
                    }
                }
            }
            catch (Exception e)
            {
            */
                foreach (List<Neuron> layer in layers)
                    foreach (Neuron neuron in layer)
                        neuron.RandomizeBias(new Random());
            //}
        }

        /// <summary>
        /// Returns the weight matrix of the network
        /// </summary>
        /// <returns>Returns the weight matrix of the network</returns>
        protected virtual List<List<List<double>>> GetWeights()
        {
            List<List<List<double>>> temp = new List<List<List<double>>>(layers.Count);
            for (int i = 1; i < layers.Count; i++)
            {
                temp.Add(new List<List<double>>(layers[i].Count));
                for (int j = 0; j < layers[i].Count; j++)
                {
                    temp[i].Add(layers[i][j].Weights);
                }
            }
            return temp;
        }

        /// <summary>
        /// Returns the bias matrix of the network
        /// </summary>
        /// <returns>Returns the bias matrix of the network</returns>
        protected virtual List<List<double>> GetBiases()
        {
            List<List<double>> temp = new List<List<double>>(layers.Count);
            for (int i = 0; i < layers.Count; i++)
            {
                temp.Add(new List<double>(layers[i].Count));
                for (int j = 0; j < layers[i].Count; j++)
                {
                    temp[i].Add(layers[i][j].Bias);
                }
            }
            return temp;
        }

        /// <summary>
        /// Gets the current number of neurons in this network
        /// </summary>
        /// <returns>Returns the count of neurons in this network</returns>
        public virtual int GetNeuralCount()
        {
            int sum = 0;
            foreach (List<Neuron> layer in layers)
                foreach (Neuron neuron in layer)
                    sum++;
            return sum;
        }

        #region Training and propagation methods
        /// <summary>
        /// Trains the network
        /// </summary>
        /// <param name="iterations">Number of iterations to go through all of the samples</param>
        /// <param name="sample_in">List of samples to use when training the network for each iteration</param>
        /// <param name="sample_out">List of expected output samples to use when training</param>
        /// <param name="errorThreshold">Threshold for which to break when the error dips below (not currently in use)</param>
        /// <param name="Reset">Flags whether to reset the network upon beginning (false)</param>
        /// <param name="delay">Millisecond delay between samples to reduce resource consumption (0)</param>
        /// <param name="RxErrEvents">Flags whether you want to receive the error value for the network in their own training updates (false)</param>
        public virtual void Train(int iterations, List<List<double>> sample_in, List<List<double>> sample_out, double errorThreshold = 0,  bool Reset = false,
            int delay = 0, bool RxErrEvents = false)
        {
            // Trains the neural network

            trainingThread = new Thread(new ThreadStart(subTrain));
            trainingThread.Start();

            void subTrain()
            {
                double Error = 0;
                TrainingUpdateEventArgs temp;
                for (int iter = 0; iter < iterations; iter++)
                {
                    // Generates the inital weight and bias tables
                    ////Console.WriteLine("Iteration: {0}", iter);

                    if (Reset)
                    {
                        GenWeightsAndBiases();
                    }

                    // Begins iterations
                    for (int i = 0; i < sample_in.Count; i++)
                    {

                        //Console.WriteLine("- Sample: {0}", i);

                        LoadSample(sample_in[i]);   // Assigns the inputs

                        ForwardPropagate(); // propagates the network forward

                        Error = BackPropagate(sample_out[i]);    // Backpropagates the network

                        // Sends all of this iteration's data back to the observers
                        temp = new TrainingUpdateEventArgs(iter, i, layers, Error, false);

                        OnTrainingUpdateEvent(temp);

                        // Stops if error is less than the threshold provided.
                        if (Error <= errorThreshold)
                            break;

                        // sleeps to free up processor space, if requested.
                        if (delay > 0)
                            Thread.Sleep(delay);
                    }

                    Error /= sample_in.Count;   // Calculate the average error of the total training session.

                    // Sends all of this iteration's data back to the observers
                    if (RxErrEvents)
                    {
                        temp = new TrainingUpdateEventArgs(iter, -1, layers, Error, false);
                        OnTrainingUpdateEvent(temp);
                    }

                    // Stops if error is less than the threshold provided.
                    if (Error <= errorThreshold)
                        break;
                }
                OnTrainingFinishEvent();    // Sends out an event notifying that training has completed.
            }
        }

        /// <summary>
        /// Forward propagates the network for the previously loaded sample
        /// </summary>
        public virtual void ForwardPropagate()
        {
            // Propagates the network forward, computes an answer

            List<Task> launchedTasks = new List<Task>(layers[0].Count);
            Predicate<Task> findActive = (Task t) => { return t.Status != TaskStatus.RanToCompletion; };

            if (!hasSubscribed)
            {
                // Subscribes to each Activation event of the Neurons
                Subscribe();
                ////Console.WriteLine("Subscribed to the neurons!");
            }

            activationCount = 0;    // Resets the activation count

            foreach (Neuron item in layers[0])
            {
                launchedTasks.Add(Task.Run(() => { item.Activate(); }));
            }

            // START HERE!!!
            while (activationCount < layers.Last().Count /*|| launchedTasks.Find(findActive) == null*/) ; // Waits until all ActivationFunction are complete or until the tasks have all ended.
        }

        /// <summary>
        /// Back propagates the network for the given expected output
        /// </summary>
        /// <param name="Sample">Expected output states</param>
        /// <returns>Returns the average error</returns>
        public virtual double BackPropagate(List<double> Sample)
        {
            // Follows the tutorial found here:
            // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
            // For help with understanding the partial derivatives look here:
            // https://sites.math.washington.edu/~aloveles/Math126Spring2014/PartialDerivativesPractice.pdf

            // ^ Is out of date, use this instead now vvv
            // http://pandamatak.com/people/anand/771/html/node37.html
            // And this one for bias back propagation
            // https://theclevermachine.wordpress.com/2014/09/06/derivation-error-backpropagation-gradient-descent-for-neural-networks/

            // ^ Is out of date, use this instead now vvv
            // XOR Example project

            // Propagates the network backward, uses computed answers, compared to real answers, to update the weights and biases
            // Returns the %error the this training sample

            // START HERE: https://youtu.be/An5z8lR8asY

            /*
            List<double> delta_K = new List<double>(layers.Last().Count);   // The deltas for the output neurons
            List<List<double>> delta_H = new List<List<double>>(layers.Count - 2);      // The deltas for the hidden neurons (excludes the input and output neurons)

            for(int i = layers.Count - 1; i > 0; i--)
            {
                // For every layer starting at the output

                if(i != layers.Count - 1)
                    delta_H.Add(new List<double>(layers[i].Count));

                for (int j = 0; j < layers[i].Count; j++)
                {
                    // For every neuron in the selected layer
                    for(int k = 0; k < layers[i][j].Weights.Count; k++)
                    {
                        // for every weight connected to that layer
                        if (i == layers.Count - 1)
                        {
                            // For the output layer use a'(netk) * (tk - Ok
                            delta_K.Add(layers[i][j].DefaultActivation.Derivate(layers[i][j].Net, layers[i][j].DefaultParameters)
                                * (Sample[j] - layers[i][j].Activation));
                        }
                        else
                        {
                            // For the hidden layers use a'(netk) * sum(weights * delta_H/K from next layer)
                            double sum = 0;

                            for(int l = 0; l < ((i == (layers.Count - 2))?delta_K.Count:delta_H[i - (layers.Count - 2) - 1].Count); l++)
                                sum += ((i == (layers.Count - 2)) ? delta_K[l] : delta_H[i - (layers.Count - 2) - 1][l]) * layers[i+1][l].Weights[j];

                            delta_H[i - (layers.Count - 2)].Add(layers[i][j].DefaultActivation.Derivate(layers[i][j].Net, layers[i][j].DefaultParameters)
                                * sum);
                        }

                        // Update the weights
                        layers[i][j].UpdateWeight()
                    }
                }
            }
            */

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                // Does the physical backpropagation
                for(int j = 0; j < layers[i].Count; j++)
                {
                    /* Variable meanings:
                         * i = current layer
                         * j = current neuron of current layer
                         */

                    if (i == layers.Count - 1)
                        layers[i][j].AssignDelta(Momentum, learningRate, Sample[j]);
                    else
                        layers[i][j].AssignDelta(Momentum, learningRate, nextLayerNeurons: layers[i + 1]);
                }
            }

            // Calculates the total error that the networkw as off by
            double ErrorTotal = 0;

            for (int i = 0; i < layers.Last().Count; i++)
                ErrorTotal += Math.Pow(Sample[i] - layers.Last()[i].Activation, 2) / 2;
            
            return ErrorTotal;
        }

        /// <summary>
        /// Method to call whenever a neuron fires in the network
        /// </summary>
        /// <param name="sender">neuron sending</param>
        /// <param name="e">activation event arguments</param>
        private void OnActiveEvent(object sender, EventArgs e)
        {
            activationCount++; // symbolizes that a neuron has fired
        }

        /// <summary>
        /// Loads a sample into the network for use in propagating
        /// </summary>
        /// <param name="Sample">Sample to load</param>
        public void LoadSample(List<double> Sample)
        {
            for (int i = 0; i < layers[0].Count; i++)
            {
                layers[0][i].RawInput = Sample[i];
            }
        }
        #endregion

        /// <summary>
        /// Subscribes the network to each neurons' activation event
        /// </summary>
        public void Subscribe()
        {
            // Causes the neural network to subscribe to all of it's neuron's activation events
            // Subscribes to each Activation event of the Neurons
            for (int i = 0; i < layers.Last().Count; i++)
                layers.Last()[i].ActiveEvent += OnActiveEvent;
        }

        #region Methods for saving a reading states.

        /// <summary>
        /// Converts this network into its xml schema
        /// </summary>
        /// <returns>The xml equivalent of this network</returns>
        protected virtual XElement GenerateFileContents()
        {
            // An overloadable method for generating the contents of the xml file.

            XElement rootTree = new XElement("Root",
                new XAttribute("LearningRate", learningRate),
                new XAttribute("Momentum", momentum));

            for (int i = 0; i < layers.Count; i++)
            {
                XElement layerTree = new XElement("Layer",
                    new XAttribute("Index", i),
                    new XAttribute("Input", layers[i][0].InputLayer),       // Is input layer?
                    new XAttribute("Output", layers[i][0].OutputLayer),     // Is output layer?
                    new XAttribute("Count", layers[i].Count));

                foreach (Neuron neuron in layers[i])
                    layerTree.Add(neuron.SerializeXml());

                rootTree.Add(layerTree);
            }

            return rootTree;
        }

        /// <summary>
        /// Loads a network from its xml schema
        /// </summary>
        /// <param name="root">XElement to load from</param>
        protected virtual void ParseFileContents(XElement root)
        {
            learningRate = Convert.ToDouble(root.Attribute("LearningRate").Value);                  // Initializes the learning rate
            momentum = Convert.ToDouble(root.Attribute("Momentum").Value);                          // Initializes the momentum

            int i = 0;
            List<List<Neuron>> temp = new List<List<Neuron>>();
            while (root.XPathSelectElement("Layer[@Index=" + i + "]") != null)
            {
                List<Neuron> temptemp = new List<Neuron>();
                XElement layer = root.XPathSelectElement("Layer[@Index=" + (i++) + "]");            // Condenses the XPath selection to a variable and increments i
                if (layer != null)
                {
                    List<XElement> neuronList = layer.XPathSelectElements("//Neuron").ToList();     // Gets the list of neurons in the layer.
                    foreach (XElement neuron in neuronList)
                    {
                        temptemp.Add(Neuron.Load(neuron));                                          // Loads each neuron
                    }
                    temp.Add(temptemp);                                                             // Loads that new layer
                }
            }
            layers = temp;                                                                          // Initializes the layer variable with the new layers
        }

        /// <summary>
        /// Saves the network to an xml file
        /// </summary>
        /// <param name="path">File directory to save to</param>
        /// <returns>Returns true on success</returns>
        public virtual bool SaveState(string path)
        {
            // Writes the current network's learning rate, momentum, and weights, and biases to an xml file.

            XElement rootTree = GenerateFileContents();

            rootTree.Save(path);

            return true;
        }

        /// <summary>
        /// Loads the network from an xml file
        /// </summary>
        /// <param name="path">file directory to load from</param>
        /// <returns>returns true on success</returns>
        public virtual bool LoadState(string path)
        {
            // Reads the current network's learning rate, momentum, and weights, and biases from an xml file.

            XElement root = XElement.Load(path);

            ParseFileContents(root);

            return true;
        }

        #endregion
        #endregion
    }
}
