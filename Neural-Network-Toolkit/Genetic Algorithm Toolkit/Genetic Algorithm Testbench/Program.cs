using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Genetic_Algorithm_Toolkit;
using Genetic_Algorithm_Toolkit.Test;

namespace Genetic_Algorithm_Testbench
{
    class Program
    {
        static void Main(string[] args)
        {
            TestCitizen tc = new TestCitizen();
            GA g = new GA(tc);
        }
    }
}
