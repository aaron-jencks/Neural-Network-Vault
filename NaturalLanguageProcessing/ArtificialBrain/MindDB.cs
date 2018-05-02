using System;
using System.Collections.Generic;
using NaturalLanguageProcessing;

namespace ArtificialBrain
{
    public class MindDB
    {
        // This class is used to control what the ai knows about its surroundings, and its word.
        // conversationMem holds important data about the current conversation.
        // longTermMem holds all of the important information that the ai knows.
        // wordDB contains every word that the ai knows.

        private Dictionary<string, object> longTermMem;
        private Dictionary<string, object> conversationMem;
        private List<Word> wordDB;

        // Accessors
        public Dictionary<string, object> LongTermMem { get => longTermMem; set => longTermMem = value; }
        public Dictionary<string, object> ConversationMem { get => conversationMem; set => conversationMem = value; }
        public List<Word> WordDB { get => wordDB; set => wordDB = value; }

        // Constructor
        public MindDB()
        {
            LongTermMem = new Dictionary<string, object>();
            ConversationMem = new Dictionary<string, object>();
            WordDB = new List<Word>();
        }

        // Public methods
        public T FindSomething<T>(string subject = "", List<string> subtopics = null)
        {
            // Searches in the dictionary for something and returns it if found, otherwise returns an empty T
            object temp;
            Dictionary<string, object> referenceDict = new Dictionary<string, object>();
            if(subject == "")
            {
                foreach(object obj in LongTermMem)
                {
                    if ((Dictionary<string, object>)obj != null)
                    {
                        if (subtopics == null && referenceDict is T)
                            return (T)obj;
                        else
                        {
                            if (subtopics != null)
                            {
                                temp = FindInDict((Dictionary<string, object>)obj, subtopics);
                                if ((T)temp != null)
                                    return (T)temp;
                                continue;
                            }
                            else
                                return (T)(new object());   // returns null
                        }
                    }
                    else
                    {
                        if (subtopics == null)
                            return (T)obj;
                        else
                            continue;
                    }
                }
                return (T)(new object());   // returns null
            }
            else
            {
                object obj = LongTermMem[subject];
                if ((Dictionary<string, object>)obj != null)
                {
                    if (subtopics == null && referenceDict is T)
                        return (T)obj;
                    else
                    {
                        if (subtopics != null)
                        {
                            temp = FindInDict((Dictionary<string, object>)obj, subtopics);
                            if ((T)temp != null)
                                return (T)temp;
                            return (T)(new object());   // returns null
                        }
                        else
                            return (T)(new object());   // returns null
                    }
                }
                else
                {
                    if (subtopics == null)
                        return (T)obj;
                    else
                        return (T)(new object());   // returns null
                }
            }

            object FindInDict(Dictionary<string, object> longTermMem, List<string> topics)
            {
                if (topics.Count > 0 && longTermMem.ContainsKey(topics[0]))
                {
                    if ((Dictionary<string, object>)longTermMem.GetValueOrDefault(topics[0]) is Dictionary<string, object>)
                    {
                        // If it finds a dictionary at the location, then it goes another level deeper.
                        string key = topics[0];
                        topics.RemoveAt(0);

                        if (topics.Count == 0)
                            return longTermMem[key];    // Return the dictionary if it's the last search term in the dictionary.
                        else
                            return FindInDict((Dictionary<string, object>)longTermMem[key], topics);    // Go one layer lower.
                    }
                    else
                    {
                        // Otherwise it returns the object it found
                        return longTermMem[topics[0]];
                    }
                }
                else
                {
                    // Otherwise return an empty object.
                    return null;
                }
            }
        }

        public Dictionary<string, object> MemoryMerge(ref MindDB mind, bool UpdateBoth = true)
        {
            // Provides functionality for merging two minds' memories together.
            foreach(string key in mind.LongTermMem.Keys)
            {
                if(longTermMem.ContainsKey(key))
                {
                    // If the key already exists
                    if (longTermMem[key] is List<object>)
                    {
                        ((List<object>)(longTermMem[key])).Add(mind.LongTermMem[key]);  // Add the object to the existing list
                    }
                    else
                        longTermMem[key] = new List<object>() { longTermMem[key], mind.LongTermMem[key] };  // Create a new list
                }
                else
                {
                    // If the key doesn't exist, add it to the dictionary
                    longTermMem.Add(key, mind.LongTermMem[key]);
                }
            }
            if(UpdateBoth)
                mind.LongTermMem = longTermMem; // Updates the other mind.

            return longTermMem; // Returns the new state of the memory
        }

        public List<Word> WordDBMerge(ref MindDB mind, bool UpdateBoth = true, bool OverwriteOld = false)
        {
            // Updates the word list of a mind, overwriting old definitions if necessary
            foreach(Word word in mind.WordDB)
            {
                if (!WordDB.Contains(word)||OverwriteOld)
                    WordDB.Add(word);
            }
            // Updates the other word if asked to.
            if (UpdateBoth)
                mind.WordDB = WordDB;

            return WordDB;
        }

        public void MindMerge(ref MindDB mind, bool UpdateBoth = true, bool OverwriteOld = false)
        {
            // Merges two minds together
            MemoryMerge(ref mind, UpdateBoth);
            WordDBMerge(ref mind, UpdateBoth, OverwriteOld);
        }

        public bool ForgetWord(Word word)
        {
            return WordDB.Remove(word);
        }

        public List<bool> ForgetWord(List<Word> words)
        {
            List<bool> temp = new List<bool>(words.Count);
            foreach (Word word in words)
                temp.Add(WordDB.Remove(word));
            return temp;
        }

        public void LearnWord(Word word)
        {
            WordDB.Add(word);
        }

        public void LearnWord(List<Word> words)
        {
            WordDB.AddRange(words);
        }

        // Public operators
        public static MindDB operator +(MindDB mind, Word word)
        {
            mind.LearnWord(word);
            return mind;
        }

        public static MindDB operator +(MindDB mind, List<Word> words)
        {
            mind.LearnWord(words);
            return mind;
        }

        public static MindDB operator +(MindDB mind, Dictionary<string, object> memory)
        {
            // Merges the mind with the given memory
            MindDB mind2 = new MindDB
            {
                LongTermMem = memory
            };
            mind.MemoryMerge(ref mind2, false);
            mind2 = null;
            return mind;
        }

        public static MindDB operator -(MindDB mind, Word word)
        {
            mind.ForgetWord(word);
            return mind;
        }

        public static MindDB operator -(MindDB mind, List<Word> words)
        {
            mind.ForgetWord(words);
            return mind;
        }
    }
}
