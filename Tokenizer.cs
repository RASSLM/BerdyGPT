using System;
using Tiktoken;
using Tiktoken.Encodings;

namespace BerdyGPT
{
    public class Tokenizer
    {
        private Encoder tokenizer = ModelToEncoder.For("gpt-4");

        public int GetVocabularySize() => 100277;

        public IReadOnlyCollection<int> Encode(string text)
        {
            return tokenizer.Encode(text);
        }

        public string Decode(IReadOnlyCollection<int> tokenIds)
        {
            return tokenizer.Decode(tokenIds);
        }
    }
}