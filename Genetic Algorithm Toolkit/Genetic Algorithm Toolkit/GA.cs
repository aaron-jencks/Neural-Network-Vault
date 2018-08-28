using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Genetic_Algorithm_Toolkit
{
    /// <summary>
    /// A parent class for a generic genetic algorithm
    /// it impliments the IGA interface
    /// 
    /// @author Aaron Jencks
    /// </summary>
    public abstract class GA : IGA
    {
        #region Properties

        #region interface properties

        public ICollection<ICitizen> Population { get; set; }

        #endregion

        /// <summary>
        /// Boolean indicator determining whether mutation should occur after crossover
        /// </summary>
        public bool DoMutation { get; set; }

        /// <summary>
        /// The percent chance (0-1) indicating how often that citizens are mutated
        /// </summary>
        public double MutationChance { get; set; }

        #endregion

        #region Methods

        #region interface methods

        public virtual ICollection<ICitizen> GeneratePopulation(Citizen CitizenTemplate, int count)
        {
            Population = new List<ICitizen>(count);
            for (int i = 0; i < count; i++)
                Population.Add(CitizenTemplate.GenerateNew());
            return Population;
        }

        public virtual ICollection<ICitizen> Crossover()
        {
            throw new NotImplementedException();
        }

        public virtual ICollection<ICitizen> Mutation()
        {
            if(DoMutation)
            {
                Random rng = new Random();

                foreach(ICitizen c in Population)
                {
                    if (rng.NextDouble() <= MutationChance)
                        Task.Factory.StartNew(c.Mutate);    // Launches in a new thread to maximize processing speed
                }
            }

            return Population;
        }

        public virtual void Selection()
        {
            foreach (ICitizen c in Population)
                Task.Factory.StartNew(c.Select);    // Launches in a new thread to maximize processing speed
        }

        #endregion

        public virtual void Train(int iterations)
        {

        }

        #endregion
    }
}
