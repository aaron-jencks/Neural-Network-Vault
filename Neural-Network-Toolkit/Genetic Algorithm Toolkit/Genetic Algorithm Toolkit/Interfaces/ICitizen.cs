using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Genetic_Algorithm_Toolkit
{
    /// <summary>
    /// A interface for the creation of custom citizens for use with
    /// the genetic algorithm (IGA) interface/parent class
    /// 
    /// @author Aaron Jencks
    /// </summary>
    public interface ICitizen
    {
        #region Properties

        /// <summary>
        /// The current fitness value for this citizen
        /// </summary>
        double Fitness { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Causes the citizen to crossover (mate) with another
        /// member of the population
        /// 
        /// Note: Also changes the passed-in mate
        /// </summary>
        /// <param name="mate">The member of the population to mate with</param>
        void Crossover(ref ICitizen mate);


        /// <summary>
        /// Causes the citizen to mutate one of its attributes
        /// </summary>
        void Mutate();

        /// <summary>
        /// This is the fitness test that the citizen will run through
        /// </summary>
        /// <returns>Returns the fitness score.</returns>
        double Select();

        #endregion
    }
}
