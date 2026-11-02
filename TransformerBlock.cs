using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BerdyGPT
{
    public class TransformerBlock : Module<Tensor, Tensor>
    {
        private MultiHeadSelfAttention selfAttention;
        private readonly int EmbeddingSize, HeadsCount, FFN_Size;
        private FeedForwardNetwork ffn;
        private LayerNorm norm1, norm2;

        public override Tensor forward(Tensor input)
        {
            var attentionOutput = selfAttention.forward(input);

            var residual1 = input + attentionOutput;

            var normalized1 = norm1.forward(residual1);

            var ffnOutput = ffn.forward(normalized1);

            var residual2 = normalized1 + ffnOutput;

            return norm2.forward(residual2);
        }

        public TransformerBlock(int embeddingSize, int headsCount, int ffnSize) : base(nameof(TransformerBlock))
        {
            EmbeddingSize = embeddingSize;
            HeadsCount = headsCount;
            FFN_Size = ffnSize;

            selfAttention = new MultiHeadSelfAttention(EmbeddingSize, HeadsCount);

            ffn = new FeedForwardNetwork(EmbeddingSize, FFN_Size);

            norm1 = LayerNorm(new long[] { EmbeddingSize });
            norm2 = LayerNorm(new long[] { EmbeddingSize });

            RegisterComponents();
        }
    }
}