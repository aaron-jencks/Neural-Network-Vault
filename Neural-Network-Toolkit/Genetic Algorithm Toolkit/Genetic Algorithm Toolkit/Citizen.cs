using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Genetic_Algorithm_Toolkit
{
    /// <summary>
    /// An abstract Class for defining the basics of a citizen of a genetic algorithm
    /// population
    /// 
    /// @author Aaron Jencks
    /// </summary>
    public abstract class Citizen : ICitizen
    {
        public abstract double Fitness { get; }

        /// <summary>
        /// Generates a new copy of the current citizen object, allows for cutsom calling
        /// in the GA class.
        /// </summary>
        /// <returns>Returns a new citizen object</returns>
        public abstract Citizen GenerateNew();

        public abstract void Crossover(ref ICitizen mate);
        public abstract void Mutate();
        public abstract double Select();
    }
}
