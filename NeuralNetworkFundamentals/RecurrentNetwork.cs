using NeuralNetworkFundamentals.Activation_Functions;
using NeuralNetworkFundamentals.Activation_Functions.Functions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using System.Xml.XPath;

namespace NeuralNetworkFundamentals
{ 
    public class RecurrentNetwork : NeuralNetwork
    {
        // This will be used to implement recurrent networks, it will handle the linking of the layers back up to themselves.

        // With recurrent layers we need to be aware of the circular reference we are making between the output of the current neuron to itself
        // This means that in the training method, we need to activate not only the input layer, but each of the recurrent layers as well
        // This will allow the recurrent layers to activate, since the recurrent hidden layer acts as an input to the recurrent neuron.
        // so when the recurrent neuron attempts to fire for the first time it needs to be marked that the input for that hidden recurrent neuron has been collected.
        // Thus, why we need to activate them during the first iteration of training, after that, every time the recurrent neuron fires, it will cause the hidden layer to fire as well.

        private List<Tuple<List<Neuron>, int>> recurrentLayers;     // Holds a list of all of the hidden recurrent networks in the network
        private List<List<int>> outputLayerConnections;             // Used for file IO, contains the lists of layers that the recurrent layers are connected to.
        private bool firstActivation;                               // Used to determine if the network has been through it's first feed-forward pass yet

        public RecurrentNetwork(List<LayerDesc> LayerInfo, List<ActivationFunction> defaultActivationFunction = null, List<ActivationParameters> Params = null,
            double learningRate = 0.5, double momentum = 0)
        {
            firstActivation = true; // Flags that the network has never been fed forward before and the hidden recurrent layers should be activated.

            Layers = new List<List<Neuron>>(LayerInfo.Count);
            outputLayerConnections = new List<List<int>>();

            if (defaultActivationFunction == null)
            {
                defaultActivationFunction = new List<ActivationFunction>(LayerInfo.Count);
                for (int i = 0; i < LayerInfo.Count; i++)
                    defaultActivationFunction.Add(new Sigmoid());
            }

            if (Params == null)
            {
                Params = new List<ActivationParameters>(LayerInfo.Count);
                for (int i = 0; i < LayerInfo.Count; i++)
                    Params.Add(new SigmoidParams());
            }

            // initializes the list of links to the recurrent layers in the network, used for indexing later on.
            recurrentLayers = new List<Tuple<List<Neuron>, int>>(LayerInfo.Count);

            // Generates the layers of Neurons
            for (int i = 0; i < LayerInfo.Count; i++)
            {
                List<Neuron> temp = new List<Neuron>(LayerInfo[i].Count);
                if (i == 0)
                    for (int j = 0; j < LayerInfo[i].Count; j++)
                        temp.Add(new Neuron(defaultActivation: defaultActivationFunction[i], defaultParameters: Params[i]));        // Creates the input layer
                else
                {
                    List<Neuron> prev = Layers[i - 1];
                    for (int j = 0; j < LayerInfo[i].Count; j++)
                        temp.Add(new Neuron(prev,
                            defaultActivation: defaultActivationFunction[i],
                            defaultParameters: Params[i],
                            outputLayer: (i == LayerInfo.Count - 1)));  // Generates the rest of the layers
                }
                Layers.Add(temp);

                // With recurrent layers we need to be aware of the circular reference we are making between the output of the current neuron to itself
                // This means that in the training method, we need to activate not only the input layer, but each of the recurrent layers as well
                // This will allow the recurrent layers to activate, since the recurrent hidden layer acts as an input to the recurrent neuron.
                // so when the recurrent neuron attempts to fire for the first time it needs to be marked that the input for that hidden recurrent neuron has been collected.
                // Thus, why we need to activate them during the first iteration of training, after that, every time the recurrent neuron fires, it will cause the hidden layer to fire as well.

                if(LayerInfo[i].Type == LayerType.recurrent)
                {
                    // Initializes a temporary item to be added to the list later.
                    Tuple<List<Neuron>, int> tempRec = new Tuple<List<Neuron>, int>(new List<Neuron>(LayerInfo[i].Count), i);

                    for (int j = 0; j < tempRec.Item2; j++)
                        tempRec.Item1.Add(new Neuron(Layers[i],
                            defaultActivation: defaultActivationFunction[i],
                            defaultParameters: Params[i]));     // Generates a list of neurons that are linked to the output of the current layer

                    if (LayerInfo[i].HasOutputs)
                    {
                        // Determines where to send the output.
                        List<int> tempOutIndex = new List<int>();
                        foreach (int index in LayerInfo[i].OutputIndex)
                        {
                            // Subscribes each output to each layer of neurons indicated in the Output Index list.
                            foreach (Neuron recNeuron in tempRec.Item1)
                                foreach (Neuron normNeuron in Layers[((index == -1) ? Layers.Count - 1 : index)])
                                    normNeuron.SubscribeToActivation(recNeuron);
                            tempOutIndex.Add(((index == -1) ? Layers.Count - 1 : index));
                        }
                        outputLayerConnections.Add(tempOutIndex);
                    }
                    else
                    {
                        // Subscribes each recurrent neuron to the original layer's neurons
                        foreach (Neuron recNeuron in tempRec.Item1)
                            foreach (Neuron normNeuron in Layers[Layers.Count - 1])
                                normNeuron.SubscribeToActivation(recNeuron);
                        outputLayerConnections.Add(new List<int>() { Layers.Count - 1 });
                    }

                    recurrentLayers.Add(tempRec);   // Adds the current layer to the archive of recurrent layers.
                }
            }
        }

        public override void Train(int iterations, List<List<double>> sample_in, List<List<double>> sample_out, double errorThreshold = 0, bool Reset = false, int delay = 0, bool RxErrEvents = false)
        {
            if (Reset)
            {
                foreach(Tuple<List<Neuron>, int> layer in recurrentLayers)
                {
                    foreach (Neuron neuron in layer.Item1)
                    {
                        // Resets all of the input values to this neuron.
                        for (int i = 0; i < neuron.InputValues.Count; i++)
                            neuron.InputValues[i] = 0;

                        // Activates teh neuron to update the value in the recurrent neuron's input.
                        neuron.Activate();
                    }
                }
                firstActivation = false;
            }

            base.Train(iterations, sample_in, sample_out, errorThreshold, Reset, delay, RxErrEvents);
        }

        public override void ForwardPropagate()
        {
            // This section of the override will take care of activating the hidden recurrent layers.
            if (firstActivation)
            {
                foreach (Tuple<List<Neuron>, int> layer in recurrentLayers)
                    foreach (Neuron neuron in layer.Item1)
                        neuron.Activate();
                firstActivation = false;
            }

            base.ForwardPropagate();
        }

        public override double BackPropagate(List<double> Sample)
        {
            double error =  base.BackPropagate(Sample);

            // Performs the back propagation on the hidden recurrent layers.
            foreach (Tuple<List<Neuron>, int> layer in recurrentLayers)
            {
                for (int j = 0; j < layer.Item1.Count; j++)
                {
                    /* Variable meanings:
                     * i = current layer
                     * j = current neuron of current layer
                     */
                        
                    layer.Item1[j].AssignDelta(Momentum, LearningRate, nextLayerNeurons: Layers[layer.Item2]);
                }
            }

            return error;
        }

        protected override void GenWeights(List<List<List<double>>> weights = null)
        {
            base.GenWeights(weights);
            foreach (Tuple<List<Neuron>, int> layer in recurrentLayers)
                foreach (Neuron neuron in layer.Item1)
                {
                    List<double> temp = new List<double>(neuron.Weights.Count);
                    for (int i = 0; i < neuron.Weights.Count; i++)
                        temp.Add(1);
                    neuron.Weights = temp;
                    // neuron.RandomizeWeights(new Random());
                }
        }

        protected override void GenBiases(List<List<double>> biases = null)
        {
            base.GenBiases(biases);
            foreach (Tuple<List<Neuron>, int> layer in recurrentLayers)
                foreach (Neuron neuron in layer.Item1)
                {
                    neuron.Bias = 0;
                    // neuron.RandomizeBias(new Random());
                }
        }

        // File IO
        protected override XElement GenerateFileContents()
        {
            XElement root =  base.GenerateFileContents();

            for (int i = 0; i < recurrentLayers.Count; i++)
            {
                XElement layerTree = new XElement("RecurrentLayer",
                    new XAttribute("Index", i),
                    new XAttribute("Input", recurrentLayers[i].Item1[0].InputLayer),        // Is input layer?
                    new XAttribute("Output", recurrentLayers[i].Item1[0].OutputLayer),      // Is output layer?
                    new XAttribute("Count", recurrentLayers[i].Item1.Count),
                    new XAttribute("LinkedInputLayer", recurrentLayers[i].Item2));          // Input layer connection

                // Serializes the output connection list
                for (int j = 0; j < outputLayerConnections.Count; j++)
                    layerTree.Add(new XElement("Output", outputLayerConnections[i]));

                foreach (Neuron neuron in recurrentLayers[i].Item1)
                    layerTree.Add(neuron.SerializeXml());

                root.Add(layerTree);
            }

            return root;
        }

        protected override void ParseFileContents(XElement root)
        {
            base.ParseFileContents(root);

            int i = 0;
            List<Tuple<List<Neuron>, int>> temp = new List<Tuple<List<Neuron>, int>>();
            List<List<int>> tempOut = new List<List<int>>();
            while (root.XPathSelectElement("RecurrentLayer[@Index=" + i + "]") != null)
            {
                List<int> outTemptemp = new List<int>();
                XElement layer = root.XPathSelectElement("RecurrentLayer[@Index=" + (i++) + "]");               // Condenses the XPath selection to a variable and increments i
                if (layer != null)
                {
                    Tuple<List<Neuron>, int> temptemp = new Tuple<List<Neuron>, int>(new List<Neuron>(),
                        Convert.ToInt32(layer.Attribute("LinkedInputLayer").Value));
                    
                    List<XElement> neuronList = layer.XPathSelectElements("//Neuron").ToList();                 // Gets the list of neurons in the layer.
                    foreach (XElement neuron in neuronList)
                    {
                        temptemp.Item1.Add(Neuron.Load(neuron));                                                // Loads each neuron
                    }
                    temp.Add(temptemp);                                                                         // Loads that new layer

                    List<XElement> outputList = layer.XPathSelectElements("//Output").ToList();                 // Gets the list of output links in the layer.
                    foreach (XElement output in outputList)
                    {
                        outTemptemp.Add(Convert.ToInt32(output.Value));
                    }
                    tempOut.Add(outTemptemp);
                }
            }
            recurrentLayers = temp;
            outputLayerConnections = tempOut;
        }
    }
}
