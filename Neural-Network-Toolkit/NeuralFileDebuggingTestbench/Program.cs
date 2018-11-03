using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Linq;
using System.Xml.Schema;
using System.Xml.XPath;
using System.Xml.Xsl;
using System.IO;
using NeuralNetworkFundamentals;

namespace NeuralFileDebuggingTestbench
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork net = new NeuralNetwork(new List<int> { 2, 2, 1 });
            string path = "XMLTest.xml";
            Console.WriteLine("Testing file write:");
            Console.WriteLine("Writing network {0} to {1}", net.ID, path);
            net.GenWeightsAndBiases();
            net.SaveState(path);
            Console.WriteLine("Successfully wrote the network to file.\nTesting file read:");
            net.LoadState(path);
            Console.WriteLine("Loaded the network from the file.");

            Console.ReadKey();
        }
    }
}
