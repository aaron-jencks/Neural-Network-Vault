using System;
using System.Collections.Generic;
using System.Linq;

namespace NaturalLanguageProcessing
{
    // This is a list of all word classes that exist, this helps determine what kind of word is being used.
    public enum WordClasses
    {
        Noun = 0, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction,
        Determiner, Exclamation
    }

    public class Word
    {
        // This class is used to define words that come from either the user, or wordnet
        // The contain the name of the word, it's word type, and it's definitions.

        // Properties
        private string name;
        private List<string> strDefinitions;
        private List<WordClasses> types;
        private List<List<DefinedWord>> definitions;
        private long id;
        private static long wordCount;

        // Accessor Methods
        public string Name { get => name; set => name = value; }
        public List<WordClasses> Types { get => types; set => types = value; }
        public List<List<DefinedWord>> Definitions { get => definitions; set => definitions = value; }
        public long ID { get => id; set => id = value; }
        public static long WordCount { get => wordCount; }
        public List<string> StrDefinitions { get => strDefinitions; set => strDefinitions = value; }

        // Constructor
        public Word(string name, List<WordClasses> types = null, List<string> strDefinitions = null, List<List<DefinedWord>> definitions = null)
        {
            this.name = name;
            this.types = types ?? new List<WordClasses>();
            this.strDefinitions = strDefinitions ?? new List<string>();
            this.definitions = definitions ?? new List<List<DefinedWord>>();
            id = wordCount++;
        }

        // Method for merging definitions of one word with another.
        public Tuple<List<WordClasses>, List<List<DefinedWord>>> DefinitionMerge(Word word)
        {
            bool found = false;
            for(int i = 0; i < word.Definitions.Count; i++)
            {
                found = false;
                // Compare all of the definitions from the other word with every definition from this word
                for (int j = 0; j < Definitions.Count; j++)
                {
                    if (Types[i] != word.Types[i])
                        continue;
                    foreach (List<DefinedWord> def in Definitions)
                    {
                        List<DefinedWord> secondNotFirst = word.Definitions[i].Except(def).ToList();
                        if (!secondNotFirst.Any())
                        {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found)
                {
                    // Add the definition if it wasn't found
                    Definitions.Add(word.Definitions[i]);
                    Types.Add(word.Types[i]);
                }
            }
            return new Tuple<List<WordClasses>, List<List<DefinedWord>>>(Types, Definitions);
        }

        // Method used for finding a definition of a word.
        public virtual void Define(int index = 0)
        {
            foreach(DefinedWord word in definitions[index])
            {
                word.Define();
            }
        }

        public static bool operator ==(Word word1, Word word2)
        {
            return word1.ID == word2.ID;
        }

        public static bool operator !=(Word word1, Word word2)
        {
            return word1.ID != word2.ID;
        }
    }
}
