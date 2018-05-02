using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetworkFundamentals.Windows_Form_Controls
{
    public partial class NetworkViewBox : Form
    {
        private NeuralNetwork net;
        private List<List<Tuple<int, int>>> neuronCoord;
        private int plotSize;

        // Accessor Methods
        public NeuralNetwork Net { get => net; set => SetupNewNetwork(ref value); }
        public int PlotSize { get => plotSize; set => plotSize = value; }

        public NetworkViewBox()
        {
            InitializeComponent();
            plotSize = 5;
            Setup();
        }

        public NetworkViewBox(ref NeuralNetwork net, int plotSize = 5)
        {
            InitializeComponent();
            this.net = net;
            this.plotSize = plotSize;
            Setup();
        }

        protected virtual void Setup()
        {
            pictureBox1.Image = new Bitmap(Width, Height);
            net = net ?? new NeuralNetwork();
            SetupNewNetwork(ref net);
            Visible = true;
        }

        public void SetupNewNetwork(ref NeuralNetwork NewNet)
        {
            if (NewNet != null)
            {
                // Updates the subscription status of the events
                if (net != null)
                    net.TrainingUpdateEvent -= TrainingUpdateEvent;
                net = NewNet;
                NewNet.TrainingUpdateEvent += TrainingUpdateEvent;

                // Clones the new network and regenerates the coordinate list
                net = NeuralNetwork.Clone(NewNet);
                neuronCoord = new List<List<Tuple<int, int>>>(net.Layers.Count);
                GenNeuronCoord();

                // Sets up the bitmap
                pictureBox1.Image = new Bitmap(Width, Height);
                pictureBox1.Width = Width;
                pictureBox1.Height = Height;
                pictureBox1.Location = new Point(0, 0);

            }
        }

        public void TrainingUpdateEvent(object sender, TrainingUpdateEventArgs e)
        {
            Task.Factory.StartNew(() =>
            {
                PaintNetworkCallback d = PaintNetwork;
                Task.Factory.StartNew(() => BeginInvoke(d, new object[] { net.Layers }));
            });  // Calls the painter on a different thread.
        }

        protected delegate void PaintNetworkCallback(List<List<Neuron>> layers);

        public virtual void PaintNetwork(List<List<Neuron>> layers)
        {
            Brush brush = new SolidBrush(Color.Black);
            Pen pen = new Pen((Brush)brush.Clone(), 1);

            lock (pictureBox1.Image)
            {
                using (Graphics g = Graphics.FromImage(pictureBox1.Image))
                {
                    g.Clear(Color.DimGray);

                    int i = 0;
                    int j = 0;

                    foreach (List<Neuron> layer in layers)
                    {
                        j = 0;
                        foreach (Neuron neuron in layer)
                        {
                            Point curNeuron = new Point(neuronCoord[i][j].Item1 + (plotSize / 2), neuronCoord[i][j].Item2 + (plotSize / 2));

                            g.DrawEllipse(pen, new Rectangle(new Point(neuronCoord[i][j].Item1, neuronCoord[i][j].Item2), new Size(plotSize, plotSize)));

                            // Draws the activation of the current neuron directly above the current plot point.
                            g.DrawString(Math.Round(neuron.Activation, 2).ToString(),
                                        new Font(Font.SystemFontName, 12), brush,
                                        curNeuron.X, curNeuron.Y - 20);

                            if ((i > 0) && (neuron.PrevWeights.Count > 0))
                            {
                                for (int k = 0; k < neuronCoord[i - 1].Count; k++)
                                {
                                    Point prevNeuron = new Point(neuronCoord[i - 1][k].Item1 + (plotSize / 2), neuronCoord[i - 1][k].Item2 + (plotSize / 2));

                                    int dX = prevNeuron.X - curNeuron.X;
                                    int dY = prevNeuron.Y - curNeuron.Y;

                                    int blue = (neuron.Weights[k] <= neuron.PrevWeights[k]) ? 255 : 0; //(int)(Math.Abs(1 - (neuron.PrevDelta - neuron.PrevDelta)) * 255);
                                    int red = (neuron.Weights[k] > neuron.PrevWeights[k]) ? 255 : 0;//(int)(Math.Abs(1 - (neuron.PrevWeights[0] - neuron.Weights[0])) * 255);
                                    int green = (int)(Math.Abs(neuron.Activation * 127));

                                    pen.Color = Color.FromArgb(255,
                                        (red > 255) ? 255 : ((red < 0) ? 0 : red),
                                        (green > 127) ? 127 : ((green < 0) ? 0 : green),
                                        (blue > 255) ? 255 : ((blue < 0) ? 0 : blue));

                                    // Draws the line that connects this neuron to each of the previous neurons via their weights.
                                    g.DrawLine(pen, curNeuron, prevNeuron);

                                    // Draws all of the weight values on their corresponding lines.
                                    g.DrawString(Math.Round(neuron.Weights[k], 2).ToString(), 
                                        new Font(Font.SystemFontName, 12), brush,
                                        curNeuron.X + (dX / 2), curNeuron.Y + (dY/2) - 20);
                                }
                            }
                            pen.Color = Color.Black;
                            j++;
                        }
                        i++;
                    }
                    pictureBox1.Invalidate();
                }
            }
        }

        protected virtual void GenNeuronCoord()
        {
            // Generates the coordinate locations for each neuron in the network.
            neuronCoord = new List<List<Tuple<int, int>>>(net.Layers.Count);
            for (int i = 0; i < net.Layers.Count; i++)
            {
                List<Tuple<int, int>> temp = new List<Tuple<int, int>>(net.Layers[i].Count);
                double scale_x = (pictureBox1.Width * 0.9)/(net.Layers.Count);
                double scale_y = (pictureBox1.Height - FontHeight)/(net.Layers[i].Count);
                double spacing_y = ((pictureBox1.Height - FontHeight) - (net.Layers[i].Count - 1) * scale_y) / 2;
                double spacing_x = (pictureBox1.Width - (net.Layers.Count - 1) * scale_x) / 2;
                for (int j = 0; j < net.Layers[i].Count; j++)
                {
                    int scaled_y = (int)(j * scale_y + spacing_y);
                    int scaled_x = (int)(i * scale_x + spacing_x);
                    temp.Add(new Tuple<int, int>(scaled_x, scaled_y));
                }
                neuronCoord.Add(temp);
            }
        }

        private void UpdateSize()
        {
            lock(pictureBox1.Image)
                pictureBox1.Image = new Bitmap(Width, Height);  // Changes the bitmap size

            pictureBox1.Width = Width;  // Changes the phsyical control size
            pictureBox1.Height = Height - FontHeight;    // Changes the physical control size
            pictureBox1.Location = new Point(0, 0); // Re-asserts the origin position

            GenNeuronCoord();   // Regenerates the coordinates.

            pictureBox1.Invalidate();   // Forces a re-draw.
        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            if (net != null)
            {
                Task.Factory.StartNew(() =>
                {
                    PaintNetworkCallback d = PaintNetwork;
                    Task.Factory.StartNew(() => BeginInvoke(d, new object[] { net.Layers }));
                });  // Calls the painter on a different thread.
            }
        }

        private void NetworkViewBox_ResizeEnd(object sender, EventArgs e)
        {
            UpdateSize();
        }

        private void NetworkViewBox_Paint(object sender, PaintEventArgs e)
        {
            UpdateSize();
        }

        private void NetworkViewBox_Resize(object sender, EventArgs e)
        {
            UpdateSize();
        }
    }
}
