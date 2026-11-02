using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tiktoken;
using static Tiktoken.Encoder;

namespace BerdyGPT
{
    public class LanguageModel : Module<Tensor, Tensor>
    {
        private Tokenizer tokenizer;
        private Embedding embedding;
        private ModuleList<TransformerBlock> blocks = new ModuleList<TransformerBlock>();
        private readonly int EmbeddingSize, HeadsCount, FFN_Size, TransformerBlocksCount;
        private LayerNorm finalNorm;
        private Linear lmHead;

        public override Tensor forward(Tensor tokensIds)
        {
            var embeddingMatrix = embedding.forward(tokensIds);

            foreach (var block in blocks)
            {
                embeddingMatrix = block.forward(embeddingMatrix);
            }

            var finalNormalization = finalNorm.forward(embeddingMatrix);

            return lmHead.forward(finalNormalization);
        }

        public LanguageModel(int embeddingSize, int headsCount, int ffnSize, int blocksCount) : base(nameof(LanguageModel))
        {
            EmbeddingSize = embeddingSize;
            HeadsCount = headsCount;
            FFN_Size = ffnSize;
            TransformerBlocksCount = blocksCount;

            tokenizer = new Tokenizer();

            embedding = Embedding(tokenizer.GetVocabularySize(), EmbeddingSize);

            for (int i = 0; i < TransformerBlocksCount; i++)
            {
                blocks.Add(new TransformerBlock(EmbeddingSize, HeadsCount, FFN_Size));
            }

            finalNorm = LayerNorm(new long[] { EmbeddingSize });
            lmHead = Linear(EmbeddingSize, tokenizer.GetVocabularySize());

            RegisterComponents();
        }
    }
}