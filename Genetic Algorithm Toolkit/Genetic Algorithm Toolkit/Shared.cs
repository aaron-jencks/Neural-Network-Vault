using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Genetic_Algorithm_Toolkit
{
    /// <summary>
    /// A container for training updates from the Genetic algorithm parent class
    /// </summary>
    public class TrainingUpdateEventArgs : EventArgs
    {
        /// <summary>
        /// Current test iteration
        /// </summary>
        public int Iteration { get; set; }

        /// <summary>
        /// Current population status
        /// </summary>
        public ICollection<ICitizen> Population { get; set; }

        public TrainingUpdateEventArgs(int iteration, ICollection<ICitizen> population)
        {
            Iteration = iteration;
            Population = population;
        }
    }
}
