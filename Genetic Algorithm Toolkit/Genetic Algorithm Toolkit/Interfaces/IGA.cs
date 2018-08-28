using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Genetic_Algorithm_Toolkit
{
    /// <summary>
    /// An interface that allows the creation of custom
    /// genetic algorithm classes
    /// 
    /// @author Aaron Jencks
    /// </summary>
    public interface IGA
    {
        #region Properties

        /// <summary>
        /// The population of this particular algorithm
        /// </summary>
        ICollection<ICitizen> Population { get; set; }

        #endregion

        #region Methods

        /// <summary>
        /// Generates a population for the genetic algorithm
        /// </summary>
        /// <param name="count">Number of citizens to generate in the population</param>
        /// <returns>Returns the population</returns>
        ICollection<ICitizen> GeneratePopulation(Citizen citizenTemplate, int count);

        /// <summary>
        /// Causes crossover (breeding) to occur in the population
        /// </summary>
        /// <returns>Returns the new population</returns>
        ICollection<ICitizen> Crossover();

        /// <summary>
        /// Causes mutation to occur in the population
        /// </summary>
        /// <returns>Returns the new population</returns>
        ICollection<ICitizen> Mutation();

        /// <summary>
        /// Runs the fitness test on the population
        /// </summary>
        void Selection();

        #endregion
    }
}
