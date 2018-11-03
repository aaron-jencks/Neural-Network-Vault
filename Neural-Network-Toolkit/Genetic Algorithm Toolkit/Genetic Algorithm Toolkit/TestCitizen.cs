using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Genetic_Algorithm_Toolkit.Test
{
    public class TestCitizen : Citizen
    {
        public override double Fitness { get; }

        /// <summary>
        /// A binary string of flags used for selection
        /// </summary>
        public List<bool> Genes { get; protected set; }

        /// <summary>
        /// Determines how many genes are in each TestCitizen when they are created.
        /// </summary>
        public static int GeneCount { get; set; } = 10;

        public TestCitizen(List<bool> genes = null, int geneCount = -1):base()
        {
            geneCount = (geneCount > 0) ? GeneCount : geneCount;

            // Initializes the genes list with default value
            Genes = genes ?? new List<bool>(geneCount);

            if(genes == null)
                for (int i = 0; i < geneCount; i++)
                    Genes.Add(false);
        }

        public override void Crossover(ref ICitizen mate)
        {
            TestCitizen realMate = (TestCitizen)mate;
            Crossover(ref realMate);
        }

        public void Crossover(ref TestCitizen mate, int slicePosA = -1, int slicePosB = -1)
        {
            Random rng = new Random();

            slicePosA = (slicePosA < 0) ? rng.Next(Genes.Count - 1) : slicePosA;
            slicePosB = (slicePosB < 0) ? rng.Next(mate.Genes.Count - 1) : slicePosB;

            // Determines if the input parameters are valid

            if (slicePosB >= mate.Genes.Count || slicePosA >= Genes.Count)
                throw new InvalidOperationException("slice position is greater than the maximum index of either the mate, or the subject!");

            // Initializes each cut List

            List<bool> cutA = new List<bool>(Genes.Count - (slicePosA + 1));
            List<bool> cutB = new List<bool>(mate.Genes.Count - (slicePosB + 1));

            // Removes the data from each subject at the splice Pos

            for (int i = 0; i < Genes.Count - (slicePosA + 1); i++)
            {
                cutA.Add(Genes[Genes.Count - 1]);
                Genes.RemoveAt(Genes.Count - 1);
            }

            cutA.Reverse();

            for (int i = 0; i < mate.Genes.Count - (slicePosB + 1); i++)
            {
                cutB.Add(mate.Genes[mate.Genes.Count - 1]);
                mate.Genes.RemoveAt(mate.Genes.Count - 1);
            }

            cutB.Reverse();

            // Adds the cut material back into the opposite subject

            foreach (bool b in cutA)
                mate.Genes.Add(b);

            foreach (bool b in cutB)
                Genes.Add(b);
        }

        /// <summary>
        /// Returns a new TestCitizen initialized to this ones current settings
        /// </summary>
        /// <returns></returns>
        public override Citizen GenerateNew()
        {
             return new TestCitizen(Genes);
        }

        public override void Mutate()
        {
            Random rng = new Random();
            int index = rng.Next(Genes.Count - 1);  // Selects a random bit
            Genes[index] = !Genes[index];           // Inverts that bit
        }

        public override double Select()
        {
            double f = 0;
            foreach (bool b in Genes)
                if (b)
                   f++;
            return f;
        }
    }
}
