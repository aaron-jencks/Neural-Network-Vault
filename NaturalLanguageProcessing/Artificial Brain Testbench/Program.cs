using ArtificialBrain;
using NaturalLanguageProcessing;
using System;
using System.Collections.Generic;

namespace Artificial_Brain_Testbench
{
    class Program
    {
        static void Main(string[] args)
        {
            MindDB mind = new MindDB();
            printf("Adding words to the dictionary...\n");
            mind += new Word("Hello", new List<WordClasses>() { WordClasses.Noun },
                new List<string>() { "an expression of greeting" });

        }

        static void printf(string message)
        {
            Console.Write(message);
        }
    }
}
