using NeuralNetworkFundamentals.Activation_Functions;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkFundamentals
{
    // Contains the definitions for the shared clusters/classes for the class files in this project

    public enum LayerType { normal = 0, recurrent, convolution }

    public class LayerDesc
    {
        // Determines what type of layer the layer of neurons is
        private int count;
        private LayerType type;
        private ActivationFunction recurrentLayerActivation;
        private ActivationParameters recurrentLayerParameters;
        private List<int> recurrentOutputLayer;

        public int Count { get => count; set => count = value; }
        public LayerType Type { get => type; set => type = value; }
        public List<int> OutputIndex { get => recurrentOutputLayer; set => recurrentOutputLayer = value; }
        public bool HasOutputs { get => (recurrentOutputLayer != null); }   // Used to determine if the output should be sent to the current layer, or other layers.
        public ActivationParameters RecurrentLayerParameters { get => recurrentLayerParameters; set => recurrentLayerParameters = value; }
        public ActivationFunction RecurrentLayerActivation { get => recurrentLayerActivation; set => recurrentLayerActivation = value; }

        // For output layer (-1: Current)
        // Leave output layer null for only current layer
        public LayerDesc(int count, LayerType type = LayerType.normal, List<int> outputLayerIndex = null,
            ActivationFunction recurrentLayerActivation = null,
            ActivationParameters recurrentLayerParameters = null)
        {
            this.count = count;
            this.type = type;
            this.recurrentLayerActivation = recurrentLayerActivation;
            this.recurrentLayerParameters = recurrentLayerParameters;
            recurrentOutputLayer = outputLayerIndex;
        }
    }
}