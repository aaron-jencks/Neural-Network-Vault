using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkFundamentals;
using NeuralNetworkFundamentals.Activation_Functions;
using NeuralNetworkFundamentals.Activation_Functions.Functions;
using static NeuralNetworkFundamentals.Neuron;

namespace NeuralNetworkFundamentals
{
    public class LSTMActivationEventArgs : EventArgs
    {
        private List<double> longTermMemory;
        private List<double> shortTermMemory;
        private NeuralNetwork forgetGate;
        private NeuralNetwork inputGate;
        private NeuralNetwork inputGate1;
        private NeuralNetwork outputGate;

        public List<double> LongTermMemory { get => longTermMemory; set => longTermMemory = value; }
        public List<double> ShortTermMemory { get => shortTermMemory; set => shortTermMemory = value; }
        public NeuralNetwork ForgetGate { get => forgetGate; set => forgetGate = value; }
        public NeuralNetwork InputGate { get => inputGate; set => inputGate = value; }
        public NeuralNetwork InputGate1 { get => inputGate1; set => inputGate1 = value; }
        public NeuralNetwork OutputGate { get => outputGate; set => outputGate = value; }

        public LSTMActivationEventArgs(List<double> longTermMemory,
            List<double> shortTermMemory,
            NeuralNetwork forgetGate,
            NeuralNetwork inputGate,
            NeuralNetwork inputGate1,
            NeuralNetwork outputGate)
        {
            this.longTermMemory = longTermMemory;
            this.shortTermMemory = shortTermMemory;
            this.forgetGate = forgetGate;
            this.inputGate = inputGate;
            this.inputGate1 = inputGate1;
            this.outputGate = outputGate;
        }
    }

    public class LSTM
    {
        public delegate void ActivateEventHandler(object sender, LSTMActivationEventArgs e);

        public event ActivateEventHandler ActivationEvent;

        public virtual void OnActivationEvent()
        {
            ActivationEvent?.Invoke(this, new LSTMActivationEventArgs(
                longTermMemory, shortTermMemory, forgetGate, inputGate, inputGate1, outputGate));
        }

        private NeuralNetwork forgetGate;       // sigmoid forget gate
        private NeuralNetwork inputGate;        // sigmoid input gate
        private NeuralNetwork inputGate1;       // tanh input gate
        private NeuralNetwork outputGate;       // sigmoid output gate

        private List<double> longTermMemory;    // The data that gets shifted back to the input, for the long term
        private List<double> shortTermMemory;   // The data that gets shifted back to the input, but only for the short term

        private List<bool> inputs_collected;    // List of booleans that represents whether or not the input for a specific neuron has been collected
        private List<double> inputs;            // List of input activations from the neurons that this cell is linked to.
        private List<long> inputIDs;            // List of input neuron ids that that this cell accepts.

        public LSTM(int inputSize, int memorySize = 1, List<Neuron> inputNeurons = null):base()
        {
            // Sets up the networks inside of the cell.
            NeuralNetwork temp = new NeuralNetwork(new List<int>() { 2 * inputSize, memorySize });
            forgetGate = temp;
            inputGate = temp;
            outputGate = new NeuralNetwork(new List<int>() { 2 * inputSize, memorySize });
            inputGate1 = new NeuralNetwork(new List<int>() { 2 * inputSize, memorySize },
                new List<ActivationFunction>() { new Tanh() },
                new List<ActivationParameters>() { new TanhParams() });

            // Initializes the memory lists for the cell
            longTermMemory = new List<double>(memorySize);
            shortTermMemory = new List<double>(memorySize);
            for (int i = 0; i < memorySize; i++)
            {
                shortTermMemory.Add(0);
                longTermMemory.Add(0);
            }

            // Sets up the flagging stuff
            inputs_collected = new List<bool>(inputSize);
            inputs = new List<double>(inputSize);
            inputIDs = new List<long>(inputSize);
            for (int i = 0; i < inputSize; i++)
            {
                inputs_collected.Add(false);
                inputs.Add(0);
                inputIDs.Add(0);
            }

            // If the input neurons are supplied, it subscribes to those neurons
            if (inputNeurons != null)
                Subscribe(inputNeurons);
        }

        public void Subscribe(List<Neuron> neurons)
        {
            foreach (Neuron neuron in neurons)
                Subscribe(neuron);
        }

        public virtual void Subscribe(Neuron neuron)
        {
            neuron.ActiveEvent += OnActivate;
            inputIDs.Add(neuron.ID);
            inputs_collected.Add(false);
            inputs.Add(0);
        }

        public virtual bool DeSubscribe(Neuron neuron)
        {
            if (inputIDs.Contains(neuron.ID))
            {
                // THIS. IS. ABSURD!!!
                long tempLong = neuron.ID;  // rEf AnD aNoNyMoUs VaLuEs ArEn'T aLlOwEd In LaMbDa ExPrEsSiOnS... -____-
                Predicate<long> idFinder = delegate (long l) { return l == tempLong; };   // Microsoft, ur mom gay
                int index = inputIDs.FindIndex(idFinder);   // (╯°□°）╯︵ ┻━┻

                neuron.ActiveEvent -= OnActivate;
                inputIDs.Remove(neuron.ID);

                // Removes the subscription from the list of inputs, collection array, etc...
                inputs_collected.RemoveAt(index);
                inputs.RemoveAt(index);

                return true;
            }
            return false;
        }

        public void OnActivate(object sender, ActivationEventArgs e)
        {
            Predicate<long> findIDMatch = (long l) => { return l == e.ID; };
            int id = inputIDs.FindIndex(findIDMatch);

            inputs[id] = e.Activation;
            inputs_collected[id] = true;

            bool temp = false;
            foreach(bool b in inputs_collected)
            {
                temp = !b;
                if (!b)
                    break;
            }

            if (!temp)
                Activate(inputs);
        }

        public virtual List<double> Activate(List<double> input)
        {
            // Loads and executes the sample in the lstm cell

            // concatenates the short term memory to the input to the cell.
            foreach (double d in shortTermMemory)
                input.Add(d);

            // TODO: MAKE THESE PARALLEL PROCESSES!!!

            // Loads the sample into the different neural layers and forwards propagates them.
            forgetGate.LoadSample(input);
            forgetGate.ForwardPropagate();

            inputGate.LoadSample(input);
            inputGate.ForwardPropagate();

            inputGate1.LoadSample(input);
            inputGate1.ForwardPropagate();

            outputGate.LoadSample(input);
            outputGate.ForwardPropagate();

            // Updates the recurrent memories

            longTermMemory = Add(
                Multiply(longTermMemory, forgetGate.Output),
                Multiply(inputGate.Output, inputGate1.Output));

            shortTermMemory = Multiply(ListTanh(longTermMemory), outputGate.Output);

            // The output is the short term memory

            OnActivationEvent();    // triggers the activation event.

            return shortTermMemory;

        }

        private List<double> Multiply(List<double> a, List<double> b)
        {
            // Mutliplies two lists together, by element.

            if (a.Count != b.Count)
                throw new Exception("Cannot multiply two lists of different sizes");

            List<double> temp = new List<double>(a.Count);

            for (int i = 0; i < a.Count; i++)
                temp.Add(a[i] * b[i]);

            return temp;
        }

        private List<double> Add(List<double> a, List<double> b)
        {
            // Adds two lists together, by element.

            if (a.Count != b.Count)
                throw new Exception("Cannot add two lists of different sizes");

            List<double> temp = new List<double>(a.Count);

            for (int i = 0; i < a.Count; i++)
                temp.Add(a[i] + b[i]);

            return temp;
        }

        private List<double> ListTanh(List<double> input)
        {
            List<double> temp = new List<double>(input.Count);
            Tanh t = new Tanh();
            foreach (double d in input)
                temp.Add(t.Activate(d, new TanhParams()));
            return temp;
        }
    }
}
