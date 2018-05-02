using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetworkFundamentals.Windows_Form_Controls
{
    public class NeuralNetworkForms
    { 
        // A class for launching various debugging windows that will be helpful for troubleshooting networks.

        private static void launchWindow(Form form)
        {
            // A reusable method for launching windows forms
            Application.EnableVisualStyles();
            Application.Run(form);
        }

        [STAThread]
        public static void ViewBox(ref NeuralNetwork net, int plotSize = 5)
        {
            //  Contains a workaround for passing a ref/out variable into a lambda expression, by creating a second method that does the same thing, but can be called synchronously
            NetworkViewBox viewBoxForm = new NetworkViewBox();
            Task.Factory.StartNew(() => launchWindow(viewBoxForm));
            viewBoxForm.SetupNewNetwork(ref net);
        }
    }
}
