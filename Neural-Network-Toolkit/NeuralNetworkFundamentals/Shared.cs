using NeuralNetworkFundamentals.Activation_Functions;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkFundamentals
{
    // Contains the definitions for the shared clusters/classes for the class files in this project
    /// <summary>
    /// Type of layer to create (normal = 0, recurrent, or convolution)
    /// </summary>
    public enum LayerType { normal = 0, recurrent, convolution }

    /// <summary>
    /// Class containing information about a layer of neurons
    /// </summary>
    public class LayerDesc
    {
        #region Properties
        // Determines what type of layer the layer of neurons is
        private int count;
        private LayerType type;
        private ActivationFunction recurrentLayerActivation;
        private ActivationParameters recurrentLayerParameters;
        private List<int> recurrentOutputLayer;
        #endregion

        #region Accessor Methods

        /// <summary>
        /// Number of neurons in this layer
        /// </summary>
        public int Count { get => count; set => count = value; }

        /// <summary>
        /// Type of layer
        /// </summary>
        public LayerType Type { get => type; set => type = value; }

        /// <summary>
        /// The index of the layer of which this layer outputs to
        /// </summary>
        public List<int> OutputIndex { get => recurrentOutputLayer; set => recurrentOutputLayer = value; }

        /// <summary>
        /// Flags if this layer has outputs
        /// </summary>
        public bool HasOutputs { get => (recurrentOutputLayer != null); }   // Used to determine if the output should be sent to the current layer, or other layers.

        /// <summary>
        /// Activation Parameters of this layer
        /// </summary>
        public ActivationParameters RecurrentLayerParameters { get => recurrentLayerParameters; set => recurrentLayerParameters = value; }

        /// <summary>
        /// Activation function of this layer
        /// </summary>
        public ActivationFunction RecurrentLayerActivation { get => recurrentLayerActivation; set => recurrentLayerActivation = value; }

        #endregion

        // For output layer (-1: Current)
        // Leave output layer null for only current layer
        /// <summary>
        /// For output layer (-1: Current)
        /// Leave output layer null for only current layer
        /// </summary>
        /// <param name="count">number of neurons in this layer</param>
        /// <param name="type">type of layer (normal, recurrent, convolutional)</param>
        /// <param name="outputLayerIndex">Where this layer outputs to</param>
        /// <param name="recurrentLayerActivation">Activation function of this layer</param>
        /// <param name="recurrentLayerParameters">Activation parameters of this layer</param>
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