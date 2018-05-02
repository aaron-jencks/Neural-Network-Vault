using System;
using System.Collections.Generic;
using System.Text;

namespace NaturalLanguageProcessing
{
    public abstract class DefinedWord
    {
        // This is a class of word that has a definition picked out, by defualt it's zero (the first one)
        // It has a method called define, that seeks out and returns the longest single lined definition it can find of it's word, using it's definition.
        private int definitionIndex;
        private Word word;

        public int DefinitionIndex { get => definitionIndex; set => definitionIndex = value; }
        public Word Word { get => word; set => word = value; }

        public DefinedWord(Word word, int index = 0)
        {
            definitionIndex = index;
            this.word = word;
        }

        public abstract void Define();
    }
}
